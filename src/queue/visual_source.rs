//! Current L0 image/page and caption source authority; v1 §5, §6.2 and §6.8.
use super::{Error, source};
use crate::{
    config::{self, Config},
    store::FileKind,
    sync::l1::{self, Work},
    treat::{Chunker, Enrichment, Router},
};
use rusqlite::{OptionalExtension, Transaction};
use std::{fs, path::Path};

/// The outbound use determines the format cap and current treatment permission.
#[derive(Clone, Copy)]
pub(crate) enum Purpose {
    Image,
    ScannedPage,
    Reference,
}

/// Verified bytes and durable file provenance; deliberately excludes Debug output.
pub(crate) struct Input {
    pub(crate) id: i64,
    pub(crate) path: String,
    pub(crate) bytes: Vec<u8>,
    pub(crate) content_hash: [u8; 32],
    pub(crate) fingerprint: [u8; 32],
}

struct Row {
    path: String,
    hash: Vec<u8>,
    size: i64,
    kind: String,
    language: Option<String>,
    fingerprint: Vec<u8>,
    max_level: i64,
    needs_vlm: bool,
    deferred: Option<String>,
}

/// Current unprotected caption text and the files that actually contributed it.
pub(crate) struct Captions {
    pub(crate) text: String,
    pub(crate) files: Vec<i64>,
    pub(crate) fingerprint: [u8; 32],
    pub(crate) truncated: bool,
}

/// Selection metadata only; source authority must be rebuilt at dispatch.
pub(crate) struct Candidate {
    pub(crate) id: i64,
    pub(crate) rank_pct: f64,
    pub(crate) referenced: bool,
}

/// Selects current image files without inventing L1 units. Referenced images
/// form the first cohort; each cohort retains the existing rank scheduling rule.
/// Caption reads are currently per image and must be measured on the image corpus.
pub(crate) fn select_images(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    work: Work<'_>,
) -> Result<Vec<Candidate>, Error> {
    if !config.llm.enabled || !config.enrich.images || config.llm.vlm_model.trim().is_empty() {
        return Ok(Vec::new());
    }
    let router = Router::new(&config.treatments, true)
        .map_err(|_| Error::Invalid("image selection policy"))?;
    let cap = config
        .limits
        .max_file_bytes()
        .map_err(|_| Error::Invalid("image file cap"))?
        .0
        .min(
            config
                .limits
                .max_image_bytes()
                .map_err(|_| Error::Invalid("image byte cap"))?
                .0,
        );
    let mut query = tx.prepare("WITH ranks AS MATERIALIZED (SELECT id,percent_rank() OVER(ORDER BY rank) AS pct FROM files)
        SELECT f.id,f.path,r.pct FROM files f JOIN ranks r ON r.id=f.id
        WHERE f.kind='image' AND f.max_level>=3 AND f.l1_pending=0
        AND f.l1_fingerprint IS NOT NULL AND f.l1_fingerprint=f.l1_desired
        AND f.l1_deferred_reason IS NULL AND f.size>=0 AND f.size<=?1 ORDER BY r.pct DESC,f.path")?;
    let mut rows = query.query([cap.min(i64::MAX as u64) as i64])?;
    let mut selected = Vec::new();
    while let Some(row) = rows.next()? {
        source::check_work(work)?;
        let path: String = row.get(1)?;
        let treatment = router.route(&path, FileKind::Image);
        if treatment.never_send
            || treatment.max_level < 3
            || !treatment.enrich.contains(&Enrichment::DescribeImage)
        {
            continue;
        }
        let references = captions(tx, root, config_root, config, &path, work)?;
        selected.push(Candidate {
            id: row.get(0)?,
            rank_pct: row.get(2)?,
            referenced: !references.files.is_empty(),
        });
    }
    // Stable partition preserves rank/path order within the two cohorts.
    selected.sort_by_key(|candidate| !candidate.referenced);
    Ok(selected)
}

/// Complete image prompt and source generation, prepared without HTTP resources.
pub(crate) struct Image {
    pub(crate) source: Input,
    pub(crate) captions: Captions,
    pub(crate) mime: &'static str,
    pub(crate) prompt: String,
    pub(crate) desired: [u8; 32],
    pub(crate) wire_fingerprint: [u8; 32],
    pub(crate) max_output_tokens: u32,
}

/// Rebuilds the image's exact desired generation for dispatch and later search
/// freshness. Raw bytes, caption policy, prompt overrides and VLM wire all count.
pub(crate) fn image(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    id: i64,
    work: Work<'_>,
) -> Result<Option<Image>, Error> {
    let Some(input) = resolve(tx, root, config_root, config, id, Purpose::Image, work)? else {
        return Ok(None);
    };
    let mime = match image::guess_format(&input.bytes) {
        Ok(image::ImageFormat::Png) => "image/png",
        Ok(image::ImageFormat::Jpeg) => "image/jpeg",
        Ok(image::ImageFormat::WebP) => "image/webp",
        Ok(image::ImageFormat::Gif) => "image/gif",
        _ => return Err(Error::Invalid("image source format")),
    };
    let captions = captions(tx, root, config_root, config, &input.path, work)?;
    let template = crate::treat::prompts::Template::load(
        config_root,
        &config.enrich.prompts_dir,
        super::Kind::DescribeImage,
    )
    .map_err(|_| Error::Invalid("image prompt override"))?;
    let mut references = captions.text.clone();
    if captions.truncated {
        references.push_str("\n[Additional referencing context omitted: prompt capacity.]\n");
    }
    let prompt = template
        .render(
            &std::collections::BTreeMap::from([("references", references.as_str())]),
            256 * 1024,
        )
        .map_err(|_| Error::Invalid("image prompt capacity or fields"))?;
    let max_output_tokens = crate::treat::prompts::output_tokens(super::Kind::DescribeImage)
        .map_err(|_| Error::Invalid("image output ceiling"))?
        .ok_or(Error::Invalid("image output ceiling"))?;
    let wire = crate::models::openai_compat::chat::image_fingerprint(
        &config.llm,
        config
            .limits
            .max_image_bytes()
            .map_err(|_| Error::Invalid("image byte cap"))?
            .0,
        &input.bytes,
        mime,
        &prompt,
        max_output_tokens,
    )
    .map_err(|_| Error::Invalid("image chat preparation"))?;
    let endpoint = crate::models::openai_compat::chat::endpoint_identity(&config.llm.url)
        .map_err(|_| Error::Invalid("image endpoint identity"))?;
    let mut proof = blake3::Hasher::new_derive_key("dkc image enrichment input v1");
    proof.update(&input.id.to_le_bytes());
    proof.update(&input.content_hash);
    proof.update(&input.fingerprint);
    proof.update(&captions.fingerprint);
    proof.update(&template.fingerprint);
    let desired =
        super::enrichment_inputs::generation(proof.finalize().as_bytes(), &wire, &endpoint);
    source::check_work(work)?;
    Ok(Some(Image {
        source: input,
        captions,
        mime,
        prompt,
        desired,
        wire_fingerprint: wire,
        max_output_tokens,
    }))
}

/// Captures referring documents in path order, with a bounded aggregate prompt.
/// Missing, changed or protected documents contribute neither text nor provenance.
pub(crate) fn captions(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    image_path: &str,
    work: Work<'_>,
) -> Result<Captions, Error> {
    const CAP: usize = 128 * 1024;
    let mut result = Captions {
        text: String::new(),
        files: Vec::new(),
        fingerprint: [0; 32],
        truncated: false,
    };
    let mut proof = blake3::Hasher::new_derive_key("dkc image caption sources v1");
    let mut query = tx.prepare(
        "SELECT id FROM files WHERE l1_pending=0 AND (
            lower(path) LIKE '%.md' OR lower(path) LIKE '%.markdown'
            OR lower(path) LIKE '%.html' OR lower(path) LIKE '%.htm') ORDER BY path",
    )?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        source::check_work(work)?;
        let Some(input) = resolve(
            tx,
            root,
            config_root,
            config,
            row.get(0)?,
            Purpose::Reference,
            work,
        )?
        else {
            continue;
        };
        let remaining = CAP.saturating_sub(result.text.len());
        if remaining <= input.path.len().saturating_add(4) {
            result.truncated = true;
            break;
        }
        let references = super::captions::extract_for(
            &input.path,
            &input.bytes,
            Some(image_path),
            remaining - input.path.len() - 4,
            work,
        )?;
        result.truncated |= references.truncated;
        if references.items.is_empty() {
            continue;
        }
        result.text.push_str(&input.path);
        result.text.push_str(":\n");
        for reference in references.items {
            source::check_work(work)?;
            result.text.push_str(&reference.context);
            result.text.push('\n');
        }
        result.text.push('\n');
        result.files.push(input.id);
        proof.update(&input.id.to_le_bytes());
        proof.update(&input.content_hash);
        proof.update(&input.fingerprint);
    }
    proof.update(&[u8::from(result.truncated)]);
    proof.update(&(result.text.len() as u64).to_le_bytes());
    proof.update(result.text.as_bytes());
    result.fingerprint = *proof.finalize().as_bytes();
    source::check_work(work)?;
    Ok(result)
}

/// Reads only admitted current file bytes, without requiring a unit or L2 vector.
/// Call again at dispatch and completion: a stored file ID is not source authority.
pub(crate) fn resolve(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    expected: &Config,
    id: i64,
    purpose: Purpose,
    work: Work<'_>,
) -> Result<Option<Input>, Error> {
    source::check_work(work)?;
    let current = config::load(config_root, None)
        .map_err(|_| Error::Invalid("visual source configuration"))?
        .config;
    if current != *expected || !current.llm.enabled {
        return Ok(None);
    }
    let root =
        fs::canonicalize(root).map_err(|_| Error::Invalid("visual source repository root"))?;
    if fs::canonicalize(config_root.join(&current.index.root))
        .ok()
        .as_ref()
        != Some(&root)
    {
        return Ok(None);
    }
    let row = tx.query_row(
        "SELECT path,blake3,size,kind,lang,l1_fingerprint,max_level,needs_vlm,l1_deferred_reason
         FROM files WHERE id=?1 AND l1_pending=0 AND l1_fingerprint IS NOT NULL
         AND l1_fingerprint=l1_desired",
        [id],
        |r| Ok(Row { path:r.get(0)?,hash:r.get(1)?,size:r.get(2)?,kind:r.get(3)?,
            language:r.get(4)?,fingerprint:r.get(5)?,max_level:r.get(6)?,
            needs_vlm:r.get(7)?,deferred:r.get(8)? }),
    ).optional()?;
    source::check_work(work)?;
    let Some(row) = row else { return Ok(None) };
    let kind = FileKind::from_db(&row.kind).ok_or(Error::Invalid("visual source kind"))?;
    let router = Router::new(&current.treatments, current.llm.enabled)
        .map_err(|_| Error::Invalid("visual source treatment"))?;
    let treatment = router.route(&row.path, kind);
    if treatment.never_send {
        return Ok(None);
    }
    let (allowed, minimum, format_cap) = match purpose {
        Purpose::Image => (
            kind == FileKind::Image
                && current.enrich.images
                && treatment.enrich.contains(&Enrichment::DescribeImage)
                && row.deferred.is_none(),
            3,
            current.limits.max_image_bytes(),
        ),
        Purpose::ScannedPage => (
            kind == FileKind::Doc
                && treatment.chunker == Chunker::Pdf
                && treatment.enrich.contains(&Enrichment::VlmPage)
                && current.enrich.scanned_pages
                && row.needs_vlm
                && matches!(
                    row.deferred.as_deref(),
                    Some("needs_vlm" | "max_pdf_pages" | "no_renderer")
                ),
            3,
            current.limits.max_doc_bytes(),
        ),
        Purpose::Reference => (
            matches!(
                Path::new(&row.path)
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(str::to_ascii_lowercase)
                    .as_deref(),
                Some("md" | "markdown" | "html" | "htm")
            ) && row.deferred.is_none(),
            1,
            current.limits.max_doc_bytes(),
        ),
    };
    if !allowed || treatment.max_level < minimum || row.max_level < minimum {
        return Ok(None);
    }
    let cap = current
        .limits
        .max_file_bytes()
        .map_err(|_| Error::Invalid("visual source file cap"))?
        .0
        .min(
            format_cap
                .map_err(|_| Error::Invalid("visual source format cap"))?
                .0,
        );
    if row.size < 0 || row.size as u64 > cap {
        return Ok(None);
    }
    let content_hash: [u8; 32] = row
        .hash
        .as_slice()
        .try_into()
        .map_err(|_| Error::Invalid("visual source hash"))?;
    let fingerprint = l1::file_fingerprint(
        &current,
        &l1::repo_name(&root, &current),
        l1::FingerprintFile {
            path: &row.path,
            content_hash: &content_hash,
            language: row.language.as_deref(),
            kind,
            capped: false,
        },
        &treatment,
    );
    if fingerprint.as_slice() != row.fingerprint {
        return Ok(None);
    }
    // No allocation proportional to catalogued size: descriptor validation and
    // byte limits precede each callback, including when a file grows during a read.
    let mut bytes = Vec::new();
    let observed = l1::observe_source(&root, &row.path, cap, work, |part| {
        bytes.extend_from_slice(part)
    });
    source::check_work(work)?;
    let Ok(observed) = observed else {
        return Ok(None);
    };
    if observed.content_capped
        || observed.size() != row.size as u64
        || observed.content_hash() != &content_hash
        || config::load(config_root, None)
            .map_err(|_| Error::Invalid("visual source configuration recheck"))?
            .config
            != current
    {
        return Ok(None);
    }
    source::check_work(work)?;
    Ok(Some(Input {
        id,
        path: row.path,
        bytes,
        content_hash,
        fingerprint,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{self, Store, StoreLimits};
    use rusqlite::params;

    #[cfg(unix)]
    use crate::sync::source::mkfifo;

    fn fixture(path: &str, bytes: &[u8], extra: &str) -> (tempfile::TempDir, Store, Config) {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        fs::write(
            root.path().join(".deepkit/config.toml"),
            format!("[index]\nrepo_name='fixture'\n[llm]\nenabled=true\n{extra}"),
        )
        .unwrap();
        fs::write(root.path().join(path), bytes).unwrap();
        let config = config::load(root.path(), None).unwrap().config;
        let store = store::open(":memory:", StoreLimits::default()).unwrap();
        let (kind, language) = crate::sync::classify_path(path);
        let treatment = Router::new(&config.treatments, true)
            .unwrap()
            .route(path, kind);
        let hash = *blake3::hash(bytes).as_bytes();
        let fingerprint = l1::file_fingerprint(
            &config,
            "fixture",
            l1::FingerprintFile {
                path,
                content_hash: &hash,
                language,
                kind,
                capped: false,
            },
            &treatment,
        );
        store
            .writer()
            .unwrap()
            .execute(
                "INSERT INTO files(id,path,blake3,size,mtime_ns,lang,kind,max_level,treatment,
                l1_pending,l1_desired,l1_fingerprint) VALUES(1,?1,?2,?3,0,?4,?5,?6,?7,0,?8,?8)",
                params![
                    path,
                    hash.as_slice(),
                    bytes.len() as i64,
                    language,
                    kind.as_str(),
                    treatment.max_level,
                    treatment.name,
                    fingerprint.as_slice()
                ],
            )
            .unwrap();
        (root, store, config)
    }
    fn admitted(root: &Path, store: &Store, config: &Config, purpose: Purpose) -> bool {
        store
            .read_snapshot(|tx| resolve(tx, root, root, config, 1, purpose, Work::default()))
            .unwrap()
            .unwrap()
            .is_some()
    }
    fn png() -> Vec<u8> {
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(2, 2)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        bytes.into_inner()
    }
    #[test]
    fn scanned_page_generation_proves_page_prompt_and_current_source() {
        let bytes = include_bytes!("../../fixtures/docs/scanned-jpeg.pdf");
        let (root, store, config) = fixture("scan.pdf", bytes, "");
        store
            .writer()
            .unwrap()
            .execute(
                "UPDATE files SET needs_vlm=1,l1_deferred_reason='needs_vlm'",
                [],
            )
            .unwrap();
        let prepare = |page| {
            store
                .read_snapshot(|tx| {
                    crate::queue::page_source::prepare(
                        tx,
                        root.path(),
                        root.path(),
                        &config,
                        1,
                        page,
                        Work::default(),
                    )
                })
                .unwrap()
        };
        let first = prepare(1).unwrap().unwrap();
        assert_eq!(first.page.number, 1);
        assert_eq!(first.page.mime, "image/jpeg");
        assert_eq!(first.max_output_tokens, 8192);
        assert_eq!(first.desired, prepare(1).unwrap().unwrap().desired);
        assert!(matches!(prepare(2), Err(Error::PagePreparation(_))));
        let prompts = root.path().join(&config.enrich.prompts_dir);
        fs::create_dir_all(&prompts).unwrap();
        fs::write(
            prompts.join("vlm_page.md"),
            "Transcribe visible text to Markdown.",
        )
        .unwrap();
        let changed = prepare(1).unwrap().unwrap();
        assert_ne!(changed.desired, first.desired);
        assert_ne!(changed.wire_fingerprint, first.wire_fingerprint);
        assert_eq!(changed.source.content_hash, first.source.content_hash);
        fs::write(root.path().join("scan.pdf"), b"changed").unwrap();
        assert!(prepare(1).unwrap().is_none());
    }
    #[test]
    fn image_authority_needs_no_unit_or_vector_and_rejects_changed_bytes() {
        let bytes = png();
        let (root, store, config) = fixture("diagram.png", &bytes, "");
        let prepared = store
            .read_snapshot(|tx| {
                resolve(
                    tx,
                    root.path(),
                    root.path(),
                    &config,
                    1,
                    Purpose::Image,
                    Work::default(),
                )
            })
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(prepared.id, 1);
        assert_eq!(prepared.path, "diagram.png");
        assert_eq!(prepared.bytes, bytes);
        assert_eq!(prepared.content_hash, *blake3::hash(&bytes).as_bytes());
        assert_ne!(prepared.fingerprint, [0; 32]);
        assert_eq!(
            store
                .writer()
                .unwrap()
                .query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        fs::write(root.path().join("diagram.png"), b"changed").unwrap();
        assert!(!admitted(root.path(), &store, &config, Purpose::Image));
    }

    #[test]
    fn image_generation_reuses_source_but_changes_with_prompt_override() {
        let (root, store, config) = fixture("diagram.png", &png(), "");
        let prepare = || {
            store
                .read_snapshot(|tx| {
                    image(tx, root.path(), root.path(), &config, 1, Work::default())
                })
                .unwrap()
                .unwrap()
                .unwrap()
        };
        let first = prepare();
        assert_eq!(first.mime, "image/png");
        assert_eq!(first.source.path, "diagram.png");
        assert!(first.captions.files.is_empty());
        assert!(first.prompt.contains("Referencing document context"));
        assert_eq!(prepare().desired, first.desired);
        let prompts = root.path().join(&config.enrich.prompts_dir);
        fs::create_dir_all(&prompts).unwrap();
        fs::write(
            prompts.join("describe_image.md"),
            "Describe visible labels.\n{{references}}",
        )
        .unwrap();
        let changed = prepare();
        assert_ne!(changed.desired, first.desired);
        assert_eq!(changed.source.content_hash, first.source.content_hash);
        assert!(changed.prompt.starts_with("Describe visible labels."));
    }
    #[test]
    fn image_caps_and_current_policy_refuse_before_source_access() {
        for extra in [
            "[limits]\nmax_file_bytes='1 B'",
            "[limits]\nmax_image_bytes='1 B'",
            "[[treatments]]\nname='private'\nmatch=['**']\nmax_level=3\nnever_send=true",
            "[[treatments]]\nname='no-enrichment'\nmatch=['**']\nmax_level=3",
        ] {
            let (root, store, config) = fixture("diagram.png", &png(), extra);
            assert!(!admitted(root.path(), &store, &config, Purpose::Image));
            // Removing the file cannot turn a cap/privacy refusal into source I/O failure.
            fs::remove_file(root.path().join("diagram.png")).unwrap();
            assert!(!admitted(root.path(), &store, &config, Purpose::Image));
        }
    }
    #[test]
    fn scanned_pdf_requires_probe_treatment_and_document_cap() {
        let bytes = include_bytes!("../../fixtures/docs/scanned.pdf");
        let (root, store, config) = fixture("scan.pdf", bytes, "");
        assert!(!admitted(
            root.path(),
            &store,
            &config,
            Purpose::ScannedPage
        ));
        store
            .writer()
            .unwrap()
            .execute(
                "UPDATE files SET needs_vlm=1,l1_deferred_reason='needs_vlm'",
                [],
            )
            .unwrap();
        assert!(admitted(root.path(), &store, &config, Purpose::ScannedPage));
        for extra in [
            "[limits]\nmax_doc_bytes='1 B'",
            "[enrich]\nscanned_pages=false",
        ] {
            let (root, store, config) = fixture("scan.pdf", bytes, extra);
            store
                .writer()
                .unwrap()
                .execute(
                    "UPDATE files SET needs_vlm=1,l1_deferred_reason='needs_vlm'",
                    [],
                )
                .unwrap();
            assert!(!admitted(
                root.path(),
                &store,
                &config,
                Purpose::ScannedPage
            ));
        }
    }
    #[test]
    fn caption_source_requires_current_unprotected_document() {
        let (root, store, config) = fixture("README.md", b"![diagram](diagram.png)", "");
        assert!(admitted(root.path(), &store, &config, Purpose::Reference));
        store
            .writer()
            .unwrap()
            .execute("UPDATE files SET l1_pending=1", [])
            .unwrap();
        assert!(!admitted(root.path(), &store, &config, Purpose::Reference));
        store
            .writer()
            .unwrap()
            .execute("UPDATE files SET l1_pending=0", [])
            .unwrap();
        fs::write(root.path().join(".deepkit/config.toml"),
            "[llm]\nenabled=true\n[[treatments]]\nname='private'\nmatch=['**']\nnever_send=true\nmax_level=1").unwrap();
        assert!(!admitted(root.path(), &store, &config, Purpose::Reference));
    }

    #[test]
    fn caption_collection_proves_only_matching_current_unprotected_sources() {
        let (root, store, config) = fixture(
            "README.md",
            b"# Architecture\n\nRequest flow ![diagram](diagram.png).",
            "",
        );
        let collect = |path: &str| {
            store
                .read_snapshot(|tx| {
                    captions(tx, root.path(), root.path(), &config, path, Work::default())
                })
                .unwrap()
                .unwrap()
        };
        let included = collect("diagram.png");
        assert_eq!(included.files, [1]);
        assert!(included.text.contains("Architecture"));
        assert!(included.text.contains("Request flow"));
        assert!(!included.truncated);
        let absent = collect("unreferenced.png");
        assert!(absent.files.is_empty() && absent.text.is_empty());
        assert_ne!(included.fingerprint, absent.fingerprint);
        fs::write(root.path().join("README.md"), b"Unindexed changed caption").unwrap();
        let stale = collect("diagram.png");
        assert!(stale.files.is_empty() && stale.text.is_empty());
        let (root, store, config) = fixture(
            "README.md",
            b"Private ![diagram](diagram.png)",
            "[[treatments]]\nname='private'\nmatch=['**']\nnever_send=true\nmax_level=1",
        );
        let protected = store
            .read_snapshot(|tx| {
                captions(
                    tx,
                    root.path(),
                    root.path(),
                    &config,
                    "diagram.png",
                    Work::default(),
                )
            })
            .unwrap()
            .unwrap();
        assert!(protected.files.is_empty() && protected.text.is_empty());
    }
    #[cfg(unix)]
    #[test]
    fn image_symlink_fifo_and_cancellation_never_grant_source_authority() {
        let (root, store, config) = fixture("diagram.png", &png(), "");
        fs::rename(
            root.path().join("diagram.png"),
            root.path().join("original.png"),
        )
        .unwrap();
        std::os::unix::fs::symlink("original.png", root.path().join("diagram.png")).unwrap();
        assert!(!admitted(root.path(), &store, &config, Purpose::Image));
        fs::remove_file(root.path().join("diagram.png")).unwrap();
        mkfifo(
            &root.path().join("diagram.png"),
            rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
        )
        .unwrap();
        assert!(!admitted(root.path(), &store, &config, Purpose::Image));
        let flag = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            store
                .read_snapshot(|tx| resolve(
                    tx,
                    root.path(),
                    root.path(),
                    &config,
                    1,
                    Purpose::Image,
                    Work {
                        cancelled: Some(&flag),
                        deadline: None
                    }
                ))
                .unwrap(),
            Err(Error::WorkerStopped)
        ));
    }
}
