//! Transaction-scoped module observations shared by map labels; v1.md §6.6–6.7/§8.
use crate::{
    config::{self, Config},
    queue::{
        Error,
        embedding::{Occurrence, Policy},
        source::{FileInput, check_work},
    },
    store::{self, FileKind, UnitKind},
    sync::{self, l1::Work},
    treat::{self, Router, context},
};
use rusqlite::{OptionalExtension, Transaction};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

struct Candidate {
    captured: store::modules::Snapshot,
    canonical: treat::module::Collated,
    breadcrumb: String,
}

/// Database-only module inputs shared within one revision-fenced map request.
/// Live discovery, source bytes, configuration and HEAD are never retained.
pub(crate) struct SnapshotCache {
    paths: BTreeSet<String>,
    snapshots: Vec<store::modules::Snapshot>,
}

/// Captures one coherent module observation; returned evidence must be checked
/// again after any repair or final response boundary. Nothing is retained here.
#[allow(
    clippy::too_many_arguments,
    reason = "explicit transaction and source authority inputs"
)]
pub(super) fn resolve(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    current: &Config,
    units: &[i64],
    minimum: i64,
    permit_protected: bool,
    work: Work<'_>,
) -> Result<BTreeMap<i64, FileInput>, Error> {
    resolve_with_cache(
        tx,
        root,
        config_root,
        current,
        units,
        minimum,
        permit_protected,
        work,
        None,
    )
}

/// Reuses database snapshots only when the caller proves the same writer
/// revision. It still performs independent live authority checks on every call.
#[allow(
    clippy::too_many_arguments,
    reason = "explicit transaction, source authority, and request cache inputs"
)]
pub(super) fn resolve_with_cache(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    current: &Config,
    units: &[i64],
    minimum: i64,
    permit_protected: bool,
    work: Work<'_>,
    mut cache: Option<&mut Option<SnapshotCache>>,
) -> Result<BTreeMap<i64, FileInput>, Error> {
    check_work(work)?;
    let empty = BTreeMap::new;
    if units.is_empty() {
        return Ok(empty());
    }
    let root = fs::canonicalize(root).map_err(|_| Error::Invalid("module repository root"))?;
    if fs::canonicalize(config_root.join(&current.index.root))
        .ok()
        .as_ref()
        != Some(&root)
        || config::reload::current(config_root)
            .map_err(|_| Error::Invalid("module configuration reload"))?
            != *current
    {
        return Ok(empty());
    }
    let head = sync::head::read(&root, work).map_err(|_| Error::Invalid("module HEAD"))?;
    let requested = units.iter().copied().collect::<BTreeSet<_>>();
    let encoded = serde_json::to_string(&units).map_err(|_| Error::Invalid("module selection"))?;
    let paths = tx
        .prepare("SELECT path FROM dirs WHERE module_unit IN (SELECT value FROM json_each(?1))")?
        .query_map([encoded], |r| r.get::<_, String>(0))?
        .collect::<Result<BTreeSet<_>, _>>()?;
    let poll = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let snapshots = match cache
        .as_ref()
        .and_then(|cached| cached.as_ref())
        .filter(|cached| paths.is_subset(&cached.paths))
    {
        Some(cached) => cached
            .snapshots
            .iter()
            .filter(|snapshot| paths.contains(&snapshot.path))
            .cloned()
            .collect(),
        None => {
            let snapshots =
                store::modules::snapshots(tx, &paths, current.enrich.modules_min_files, &poll)?;
            if let Some(cache) = cache.as_mut() {
                **cache = Some(SnapshotCache {
                    paths,
                    snapshots: snapshots.clone(),
                });
            }
            snapshots
        }
    };
    let name = sync::l1::repo_name(&root, current);
    let mut candidates = BTreeMap::new();
    let mut sources = BTreeMap::new();
    let mut scope = BTreeSet::new();
    for captured in snapshots {
        check_work(work)?;
        let Some(unit) = captured.unit.filter(|id| requested.contains(id)) else {
            continue;
        };
        if !store::modules::sources_match(tx, unit, &captured)? {
            continue;
        }
        let canonical = treat::module::collate(&captured.input);
        let breadcrumb = context::breadcrumb(
            context::ContextFile {
                repo_name: &name,
                path: &captured.path,
            },
            None,
            &[],
            true,
        );
        let key = format!("module:{}", captured.path);
        let valid = tx.query_row("SELECT u.content_hash=?2 AND u.scope_path=?3 AND u.header=?4 AND c.text=?5 AND u.source_key=?6 AND u.parent_unit IS NULL FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.id=?1 AND u.kind='module' AND u.file_id IS NULL AND u.ladder_level>=1", rusqlite::params![unit,canonical.hash.as_slice(),breadcrumb,canonical.header,canonical.text,key], |r| r.get::<_, bool>(0)).optional()?.unwrap_or(false);
        if !valid {
            continue;
        }
        scope.insert(captured.path.clone());
        for source in &captured.sources {
            scope.insert(source.path.clone());
            sources.insert(source.id, source.clone());
        }
        candidates.insert(
            unit,
            Candidate {
                captured,
                canonical,
                breadcrumb,
            },
        );
    }
    if candidates.is_empty() {
        return Ok(empty());
    }
    let source_paths = sources
        .values()
        .map(|s| s.path.clone())
        .collect::<BTreeSet<_>>();
    let census = || -> Result<Option<BTreeSet<String>>, Error> {
        let discovered = sync::discovery::discover_paths(&root, &scope, source_paths.len(), work);
        check_work(work)?;
        let Ok(discovered) = discovered else {
            return Ok(None);
        };
        if discovered.max_files_reached
            || discovered.skipped_non_unicode_paths != 0
            || discovered.head != head
        {
            return Ok(None);
        }
        Ok(Some(discovered.paths.into_iter().collect()))
    };
    let Some(before) = census()? else {
        return Ok(empty());
    };
    let router = Router::new(&current.treatments, current.llm.enabled)
        .map_err(|_| Error::Invalid("module treatment policy"))?;
    let maximum = current
        .limits
        .max_file_bytes()
        .map_err(|_| Error::Invalid("module source cap"))?
        .0;
    let mut verified = BTreeMap::new();
    for source in sources.values() {
        check_work(work)?;
        if !before.contains(&source.path) {
            continue;
        }
        let observed = sync::l1::observe_source(&root, &source.path, maximum, work, |_| {});
        check_work(work)?;
        let Ok(observed) = observed else { continue };
        if source.size < 0
            || observed.size() != source.size as u64
            || observed.content_hash().as_slice() != source.hash
        {
            continue;
        }
        let kind = FileKind::from_db(&source.kind).ok_or(Error::Invalid("module source kind"))?;
        let treatment = router.route(&source.path, kind);
        if source.max_level >= 1 {
            let hash: [u8; 32] = source
                .hash
                .as_slice()
                .try_into()
                .map_err(|_| Error::Invalid("module source hash"))?;
            let expected = sync::l1::file_fingerprint(
                current,
                &name,
                sync::l1::FingerprintFile {
                    path: &source.path,
                    content_hash: &hash,
                    language: source.language.as_deref(),
                    kind,
                    capped: observed.content_capped,
                },
                &treatment,
            );
            if source.fingerprint.as_deref() != Some(expected.as_slice()) {
                continue;
            }
        }
        verified.insert(source.id, (treatment.max_level, treatment.never_send));
    }
    let Some(after) = census()? else {
        return Ok(empty());
    };
    if config::reload::current(config_root)
        .map_err(|_| Error::Invalid("module configuration reload"))?
        != *current
        || sync::head::read(&root, work).map_err(|_| Error::Invalid("module HEAD"))? != head
    {
        return Ok(empty());
    }
    let settings = sync::l1::settings_fingerprint(current);
    let mut result = BTreeMap::new();
    for (unit, candidate) in candidates {
        check_work(work)?;
        let wanted = candidate
            .captured
            .input
            .files
            .iter()
            .map(|file| {
                if candidate.captured.path.is_empty() {
                    file.path.clone()
                } else {
                    format!("{}/{}", candidate.captured.path, file.path)
                }
            })
            .collect::<BTreeSet<_>>();
        let inside = |path: &&String| Path::new(path).starts_with(&candidate.captured.path);
        if before
            .iter()
            .filter(inside)
            .cloned()
            .collect::<BTreeSet<_>>()
            != wanted
            || after
                .iter()
                .filter(inside)
                .cloned()
                .collect::<BTreeSet<_>>()
                != wanted
        {
            continue;
        }
        let mut max_level = 0;
        let mut never_send = false;
        let mut valid = true;
        let mut fingerprint = blake3::Hasher::new_derive_key("dkc module source policy v1");
        fingerprint.update(&settings);
        for source in &candidate.captured.sources {
            check_work(work)?;
            let Some(&(level, protected)) = verified
                .get(&source.id)
                .filter(|_| after.contains(&source.path))
            else {
                valid = false;
                break;
            };
            max_level = max_level.max(level);
            never_send |= protected;
            for bytes in [source.path.as_bytes(), source.hash.as_slice()] {
                fingerprint.update(&(bytes.len() as u64).to_le_bytes());
                fingerprint.update(bytes);
            }
            fingerprint.update(&level.to_le_bytes());
            fingerprint.update(&[u8::from(protected)]);
        }
        if !valid || max_level < minimum || (never_send && !permit_protected) {
            continue;
        }
        result.insert(
            unit,
            FileInput {
                occurrence: Occurrence {
                    id: unit,
                    content_hash: candidate.canonical.hash,
                    kind: UnitKind::Module,
                },
                context: context::history_input(
                    &candidate.breadcrumb,
                    &candidate.canonical.header,
                    "module",
                    &candidate.canonical.text,
                    context::Options::from(&current.embedder),
                ),
                policy: Policy {
                    fingerprint: *fingerprint.finalize().as_bytes(),
                    max_level,
                    never_send,
                },
            },
        );
    }
    check_work(work)?;
    Ok(result)
}
