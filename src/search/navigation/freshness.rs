//! Bounded verification of returned source locations; `v1.md` §4.6 and §9.
use super::{Result, Store, source};
use crate::sync::l1::Work;
use serde_json::Value;
use std::{io::Read, path::Path, time::SystemTime};

/// Request-scoped L1 identity inputs shared by every selected source proof.
pub(crate) struct L1Verifier<'a> {
    settings: &'a crate::config::Config,
    repo_name: String,
    router: crate::treat::Router,
}

impl<'a> L1Verifier<'a> {
    /// Compiles treatment routing once for a complete public-tool invocation.
    pub(crate) fn new(root: &Path, settings: &'a crate::config::Config) -> Option<Self> {
        Some(Self {
            settings,
            repo_name: crate::sync::l1::repo_name(root, settings),
            router: crate::treat::Router::new(&settings.treatments, settings.llm.enabled).ok()?,
        })
    }

    fn current(&self, path: Option<&str>, source: &crate::store::OpenSource) -> bool {
        super::continuation::current_l1_with(
            self.settings,
            path,
            source,
            &self.repo_name,
            &self.router,
        )
    }
}

/// Checks a hit against its completed occurrence generation and current source.
/// Refused hits must be suppressed, not returned with potentially invalid ranges.
pub(crate) fn verify_hit(
    store: &Store,
    root: &Path,
    hit: &Value,
    max_file_bytes: u64,
    work: Work<'_>,
) -> Result<std::result::Result<(), &'static str>> {
    Ok(verify_hit_age(store, root, hit, max_file_bytes, work)?.map(|_| ()))
}

/// Verifies the source and reports seconds since its filesystem modification time.
/// Missing timestamps remain unknown; future timestamps clamp to zero.
pub(crate) fn verify_hit_age(
    store: &Store,
    root: &Path,
    hit: &Value,
    max_file_bytes: u64,
    work: Work<'_>,
) -> Result<std::result::Result<Option<u64>, &'static str>> {
    Ok(hit_authority(store, root, hit, max_file_bytes, work)?.map(|(age, _)| age))
}

/// Returns the completed source observation that was checked against the bytes.
/// Retaining it prevents final validation from accepting a different publication.
pub(crate) fn hit_authority(
    store: &Store,
    root: &Path,
    hit: &Value,
    max_file_bytes: u64,
    work: Work<'_>,
) -> Result<std::result::Result<(Option<u64>, crate::store::OpenSource), &'static str>> {
    verify_hit_source(store, root, hit, max_file_bytes, work, None)
}

fn changed_age(modified: Option<SystemTime>, now: SystemTime) -> Option<u64> {
    modified.map(|modified| now.duration_since(modified).unwrap_or_default().as_secs())
}

fn verify_hit_source(
    store: &Store,
    root: &Path,
    hit: &Value,
    max_file_bytes: u64,
    work: Work<'_>,
    source_bytes: Option<&mut Vec<u8>>,
) -> Result<std::result::Result<(Option<u64>, crate::store::OpenSource), &'static str>> {
    if work.check().is_err() {
        return Ok(Err("timeout"));
    }
    let (Some(id), Some(_path)) = (hit["id"].as_str(), hit["path"].as_str()) else {
        return Ok(Err("source_unavailable"));
    };
    let Some(before) = store.open_source_checked(id, &|| {
        work.check()
            .map_err(|_| crate::store::Error::L1InputsChanged)
    })?
    else {
        return Ok(Err("l1_pending_or_unavailable"));
    };
    let age = match verify_file_hit_source(root, hit, &before, max_file_bytes, work, source_bytes) {
        Ok(age) => age,
        Err(reason) => return Ok(Err(reason)),
    };
    if store
        .open_source_checked(id, &|| {
            work.check()
                .map_err(|_| crate::store::Error::L1InputsChanged)
        })?
        .as_ref()
        != Some(&before)
    {
        return Ok(Err("l1_pending_or_unavailable"));
    }
    Ok(Ok((age, before)))
}

/// Proves current file bytes and ranges against an already observed L1 source.
fn verify_file_hit_source(
    root: &Path,
    hit: &Value,
    before: &crate::store::OpenSource,
    max_file_bytes: u64,
    work: Work<'_>,
    mut source_bytes: Option<&mut Vec<u8>>,
) -> std::result::Result<Option<u64>, &'static str> {
    if work.check().is_err() {
        return Err("timeout");
    }
    let Some(path) = hit["path"].as_str() else {
        return Err("source_unavailable");
    };
    let Ok(mut file) = source::open_regular(root, Path::new(path)) else {
        return Err("source_unavailable");
    };
    let Ok(metadata) = file.metadata() else {
        return Err("source_unavailable");
    };
    if metadata.len() > max_file_bytes {
        return Err("source_bytes");
    }
    let mut hash = blake3::Hasher::new();
    let mut buffer = [0_u8; 16 * 1024];
    let mut bytes = 0_u64;
    let mut lines = 0_u64;
    let mut last = None;
    loop {
        if work.check().is_err() {
            return Err("timeout");
        }
        let Ok(count) = file.read(&mut buffer) else {
            return Err("source_unavailable");
        };
        if count == 0 {
            break;
        }
        bytes = bytes.saturating_add(count as u64);
        if bytes > max_file_bytes {
            return Err("source_bytes");
        }
        hash.update(&buffer[..count]);
        if let Some(source) = source_bytes.as_deref_mut() {
            source.extend_from_slice(&buffer[..count]);
        }
        lines += buffer[..count]
            .iter()
            .filter(|byte| **byte == b'\n')
            .count() as u64;
        last = Some(buffer[count - 1]);
    }
    lines += u64::from(last.is_some_and(|byte| byte != b'\n'));
    if hash.finalize().as_bytes().as_slice() != before.hash {
        return Err("source_changed");
    }
    // Document chunk positions (notably notebook cells) belong to extracted
    // text, not the encoded source. The byte hash and L1 fence still apply.
    if before.kind == "code" && matches!(hit["kind"].as_str(), Some("chunk" | "symbol")) {
        let valid = hit["lines"][0]
            .as_u64()
            .zip(hit["lines"][1].as_u64())
            .is_some_and(|(start, end)| start > 0 && end >= start && end <= lines);
        if !valid {
            return Err("invalid_source_range");
        }
    }
    Ok(changed_age(metadata.modified().ok(), SystemTime::now()))
}

/// Checks all returned lines in one reference group against its current file.
/// The unit anchors the completed file generation; references need not fall in
/// that unit's own range (graph edges can point elsewhere in the same file).
#[cfg(test)]
pub(crate) fn verify_reference_group(
    store: &Store,
    root: &Path,
    settings: &crate::config::Config,
    group: &super::ReferenceGroup,
    max_file_bytes: u64,
    work: Work<'_>,
) -> Result<std::result::Result<(), &'static str>> {
    let Some(verifier) = L1Verifier::new(root, settings) else {
        return Ok(Err("l1_config_stale"));
    };
    Ok(reference_authority_with(store, root, group, max_file_bytes, work, &verifier)?.map(|_| ()))
}

/// Verifies one reference group with request-scoped L1 identity inputs.
pub(crate) fn reference_authority_with(
    store: &Store,
    root: &Path,
    group: &super::ReferenceGroup,
    max_file_bytes: u64,
    work: Work<'_>,
    verifier: &L1Verifier<'_>,
) -> Result<std::result::Result<(crate::store::StoredUnit, crate::store::OpenSource), &'static str>>
{
    if work.check().is_err() {
        return Ok(Err("timeout"));
    }
    let Some(unit) = store.unit_by_path_checked(&group.path, None, None, &|| {
        work.check()
            .map_err(|_| crate::store::Error::L1InputsChanged)
    })?
    else {
        return Ok(Err("l1_pending_or_unavailable"));
    };
    let Some(authority) = store.open_source_checked(&super::public_id(&unit), &|| {
        work.check()
            .map_err(|_| crate::store::Error::L1InputsChanged)
    })?
    else {
        return Ok(Err("l1_pending_or_unavailable"));
    };
    // Unchanged source bytes do not establish current tokenization, chunking,
    // context or treatment authority after a configuration reload.
    if !verifier.current(Some(&group.path), &authority) {
        return Ok(Err("l1_config_stale"));
    }
    let hit = serde_json::json!({
        "id": super::public_id(&unit), "path": group.path, "kind": "chunk",
        "lines": [group.lines.first(), group.lines.last()]
    });
    Ok(verify_hit(store, root, &hit, max_file_bytes, work)?.map(|()| (unit, authority)))
}

/// Prior source observation and database revision for a final live-byte check.
pub(crate) struct ReferenceProof<'a> {
    pub(crate) unit: &'a crate::store::StoredUnit,
    pub(crate) authority: &'a crate::store::OpenSource,
    pub(crate) revision: crate::store::modules::Revision,
}

/// Rechecks live bytes under the prior database revision. `None` requests a
/// fresh full proof after a local or external database write.
pub(crate) fn reference_authority_cached_with(
    store: &Store,
    root: &Path,
    group: &super::ReferenceGroup,
    proof: ReferenceProof<'_>,
    max_file_bytes: u64,
    work: Work<'_>,
    verifier: &L1Verifier<'_>,
) -> Result<
    Option<std::result::Result<(crate::store::StoredUnit, crate::store::OpenSource), &'static str>>,
> {
    let check = || {
        work.check()
            .map_err(|_| crate::store::Error::L1InputsChanged)
    };
    if store.revision_checked(&check)? != proof.revision {
        return Ok(None);
    }
    let verified = if !verifier.current(Some(&group.path), proof.authority) {
        Err("l1_config_stale")
    } else {
        let hit = serde_json::json!({
            "path": group.path, "kind": "chunk",
            "lines": [group.lines.first(), group.lines.last()]
        });
        verify_file_hit_source(root, &hit, proof.authority, max_file_bytes, work, None)
            .map(|_| (proof.unit.clone(), proof.authority.clone()))
    };
    if store.revision_checked(&check)? != proof.revision {
        return Ok(None);
    }
    Ok(Some(verified))
}

/// Verifies a symbol range and derives documentation from the same hashed bytes.
/// No pathname reread may introduce unverified comments or block on special files.
#[cfg(test)]
pub(crate) fn verify_symbol(
    store: &Store,
    root: &Path,
    settings: &crate::config::Config,
    symbol: &mut super::SymbolResult,
    max_file_bytes: u64,
    work: Work<'_>,
) -> Result<()> {
    let verifier = L1Verifier::new(root, settings).ok_or(super::Error::OpenSourceUnavailable)?;
    verify_symbol_with(store, root, symbol, max_file_bytes, work, &verifier)
}

/// Verifies one symbol with request-scoped L1 identity inputs.
pub(crate) fn verify_symbol_with(
    store: &Store,
    root: &Path,
    symbol: &mut super::SymbolResult,
    max_file_bytes: u64,
    work: Work<'_>,
    verifier: &L1Verifier<'_>,
) -> Result<()> {
    work.check().map_err(|_| super::Error::Deadline)?;
    let check = || {
        work.check()
            .map_err(|_| crate::store::Error::L1InputsChanged)
    };
    let unit = match symbol.selected.unit_id {
        Some(id) => store.unit_by_id_checked(id, &check)?,
        None => store.unit_by_path_checked(&symbol.path, None, None, &check)?,
    }
    .filter(|unit| unit.path.as_deref() == Some(symbol.path.as_str()))
    .ok_or(super::Error::OpenSourceUnavailable)?;
    let authority = store
        .open_source_checked(&super::public_id(&unit), &|| {
            work.check()
                .map_err(|_| crate::store::Error::L1InputsChanged)
        })?
        .ok_or(super::Error::OpenSourceUnavailable)?;
    if !verifier.current(Some(&symbol.path), &authority) {
        return Err(super::Error::OpenSourceUnavailable);
    }
    let hit = serde_json::json!({
        "id": super::public_id(&unit), "path": symbol.path, "kind": "symbol",
        "lines": symbol.lines
    });
    let mut bytes = Vec::new();
    match verify_hit_source(store, root, &hit, max_file_bytes, work, Some(&mut bytes))? {
        Ok(_) => {}
        Err("timeout") => return Err(super::Error::Deadline),
        Err(_) => return Err(super::Error::OpenSourceUnavailable),
    }
    work.check().map_err(|_| super::Error::Deadline)?;
    let source = std::str::from_utf8(&bytes).map_err(|_| super::Error::OpenSourceUnavailable)?;
    symbol.doc = super::symbol_doc_comment(source, symbol.lines[0]);
    symbol.evidence = Some((unit, authority));
    Ok(())
}

/// Verifies a map file once, including every admitted signature range.
/// Occurrence-backed files require canonical current-config L1 even when the
/// budget omits signatures. Catalogue-only files require a completed, current
/// source/config identity; capped digests grant only metadata-level authority.
pub(crate) fn verify_map_file(
    store: &Store,
    root: &Path,
    settings: &crate::config::Config,
    path: &str,
    symbols: &[crate::store::ToolSymbol],
    max_file_bytes: u64,
    work: Work<'_>,
) -> Result<std::result::Result<(), &'static str>> {
    Ok(map_authority(store, root, settings, path, symbols, max_file_bytes, work)?.map(|_| ()))
}

/// Completed per-file evidence retained until final map validation. Ladder-only
/// progress is excluded from occurrence identity, as it does not change L1.
#[derive(Debug, PartialEq)]
pub(crate) enum MapAuthority {
    Occurrences {
        source: crate::store::OpenSource,
        units: Vec<(crate::store::StoredUnit, crate::store::OpenSource)>,
    },
    Catalogue {
        source: Box<crate::store::CatalogueSource>,
        observed: crate::sync::ScannedFile,
    },
}

/// Verifies and retains the source plus every displayed symbol's occurrence.
pub(crate) fn map_authority(
    store: &Store,
    root: &Path,
    settings: &crate::config::Config,
    path: &str,
    symbols: &[crate::store::ToolSymbol],
    max_file_bytes: u64,
    work: Work<'_>,
) -> Result<std::result::Result<MapAuthority, &'static str>> {
    work.check().map_err(|_| super::Error::Deadline)?;
    let check = || {
        work.check()
            .map_err(|_| crate::store::Error::L1InputsChanged)
    };
    let Some(unit) = store.unit_by_path_checked(path, None, None, &check)? else {
        if !symbols.is_empty() {
            return Ok(Err("l1_pending_or_unavailable"));
        }
        return verify_catalogue_file(store, root, settings, path, max_file_bytes, work);
    };
    let id = super::public_id(&unit);
    let Some(authority) = store.open_source_checked(&id, &check)? else {
        return Ok(Err("l1_pending_or_unavailable"));
    };
    if !super::continuation::current_l1(root, settings, Some(path), &authority) {
        return Ok(Err("l1_config_stale"));
    }
    if symbols
        .iter()
        .any(|symbol| symbol.line_start <= 0 || symbol.line_end < symbol.line_start)
    {
        return Ok(Err("invalid_source_range"));
    }
    let hit = serde_json::json!({
        "id": id, "path": path,
        "kind": if symbols.is_empty() { "file" } else { "symbol" },
        "lines": [symbols.iter().map(|s| s.line_start).min(), symbols.iter().map(|s| s.line_end).max()]
    });
    let verified = verify_hit(store, root, &hit, max_file_bytes, work)?;
    // The hash/range check must use the same completion whose config was checked.
    if store.open_source_checked(&id, &check)?.as_ref() != Some(&authority) {
        return Ok(Err("l1_pending_or_unavailable"));
    }
    if let Err(reason) = verified {
        return Ok(Err(reason));
    }
    let mut units = Vec::new();
    let anchor = unit;
    for symbol in symbols {
        work.check().map_err(|_| super::Error::Deadline)?;
        let Some(id) = symbol.unit_id else {
            return Ok(Err("l1_pending_or_unavailable"));
        };
        let Some(unit) = store.unit_by_id_checked(id, &check)? else {
            return Ok(Err("l1_pending_or_unavailable"));
        };
        if unit.path.as_deref() != Some(path) {
            return Ok(Err("occurrence_changed"));
        }
        let Some(source) = store.open_source_checked(&super::public_id(&unit), &check)? else {
            return Ok(Err("l1_pending_or_unavailable"));
        };
        if source.hash != authority.hash || source.fingerprint != authority.fingerprint {
            return Ok(Err("occurrence_changed"));
        }
        units.push((unit, source));
    }
    let anchor_source = store.open_source_checked(&id, &check)?;
    if anchor_source.as_ref() != Some(&authority) {
        return Ok(Err("occurrence_changed"));
    }
    let Some(anchor_source) = anchor_source else {
        return Ok(Err("occurrence_changed"));
    };
    units.push((anchor, anchor_source));
    for (unit, source) in &units {
        if store
            .open_source_checked(&super::public_id(unit), &check)?
            .as_ref()
            != Some(source)
        {
            return Ok(Err("occurrence_changed"));
        }
    }
    let current = store.checked_read_snapshot(&check, |snapshot| {
        for (unit, _) in &units {
            check()?;
            if !crate::store::occurrence_matches_in(snapshot, unit)? {
                return Ok::<_, crate::store::Error>(false);
            }
        }
        Ok(true)
    })??;
    if !current || store.open_source_checked(&id, &check)?.as_ref() != Some(&authority) {
        return Ok(Err("occurrence_changed"));
    }
    for (unit, _) in &mut units {
        unit.ladder_level = 0;
    }
    Ok(Ok(MapAuthority::Occurrences {
        source: authority,
        units,
    }))
}

fn verify_catalogue_file(
    store: &Store,
    root: &Path,
    settings: &crate::config::Config,
    path: &str,
    max_file_bytes: u64,
    work: Work<'_>,
) -> Result<std::result::Result<MapAuthority, &'static str>> {
    use crate::sync::l1;
    let check = || {
        work.check()
            .map_err(|_| crate::store::Error::L1InputsChanged)
    };
    let Some(before) = store.catalogue_source_checked(path, &check)? else {
        return Ok(Err("l1_pending_or_unavailable"));
    };
    if before.pending
        || before.has_occurrences
        || before.deferred.is_some()
        || before.completed.is_none()
        || before.completed != before.desired
    {
        return Ok(Err("l1_pending_or_unavailable"));
    }
    let observed = l1::observe_source(root, path, max_file_bytes, work, |_| {});
    work.check().map_err(|_| super::Error::Deadline)?;
    let Ok(observed) = observed else {
        return Ok(Err("source_unavailable"));
    };
    if observed.content_hash().as_slice() != before.file.blake3
        || i64::try_from(observed.size()).ok() != Some(before.file.size)
        || observed.kind.as_str() != before.file.kind
        || observed.lang != before.file.lang
    {
        return Ok(Err("source_changed"));
    }
    let Ok(router) = crate::treat::Router::new(&settings.treatments, settings.llm.enabled) else {
        return Ok(Err("l1_config_stale"));
    };
    let treatment = router.route(path, observed.kind);
    let fingerprint = l1::file_fingerprint(
        settings,
        &l1::repo_name(root, settings),
        l1::FingerprintFile {
            path,
            content_hash: observed.content_hash(),
            language: observed.lang.as_deref(),
            kind: observed.kind,
            capped: observed.content_capped,
        },
        &treatment,
    );
    if before.completed.as_deref() != Some(fingerprint.as_slice())
        || treatment.max_level != before.file.max_level
        || treatment.name != before.file.treatment
    {
        return Ok(Err("l1_config_stale"));
    }
    let verified = l1::verify_source(root, &observed, max_file_bytes, work);
    work.check().map_err(|_| super::Error::Deadline)?;
    if verified.is_err() {
        return Ok(Err("source_changed"));
    }
    if store.catalogue_source_checked(path, &check)?.as_ref() != Some(&before) {
        return Ok(Err("l1_pending_or_unavailable"));
    }
    Ok(Ok(MapAuthority::Catalogue {
        source: Box::new(before),
        observed,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn catalogue_capped_digest_is_metadata_only_and_never_reused_after_cap_change() {
        use crate::sync::{self, l1};
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("large.rs"), "pub fn large() {}\n").unwrap();
        let store = crate::store::open(Path::new(":memory:"), Default::default()).unwrap();
        let mut settings = crate::config::Config::default();
        settings.limits.max_file_bytes = "4 B".into();
        let scan = sync::scan(
            root.path(),
            sync::SyncLimits {
                max_file_bytes: 4,
                ..Default::default()
            },
        )
        .unwrap();
        let publication = l1::publish(
            &store,
            l1::PublishInput {
                work: Work::default(),
                scan: &scan,
                git: None,
                snapshot: &store.index_snapshot().unwrap(),
                generation: store.l1_generation().unwrap(),
                root: root.path(),
                settings: &settings,
                full: false,
            },
        )
        .unwrap();
        let file = scan.file("large.rs").unwrap();
        let row = store.index_snapshot().unwrap().files.remove(0);
        l1::Processor::new(root.path(), &settings)
            .unwrap()
            .process_file(
                &store,
                l1::FileInput {
                    work: Work::default(),
                    file,
                    fence: crate::store::l1::Fence {
                        generation: publication.generation,
                        file_id: row.id,
                        path: "large.rs",
                        content_hash: file.content_hash(),
                        fingerprint: &publication.fingerprints["large.rs"],
                    },
                },
                || Ok(()),
            )
            .unwrap();
        let before = store
            .catalogue_source_checked("large.rs", &|| Ok(()))
            .unwrap()
            .unwrap();
        assert!(!before.has_occurrences);
        assert!(before.deferred.is_none());
        assert_eq!(
            verify_map_file(
                &store,
                root.path(),
                &settings,
                "large.rs",
                &[],
                4,
                Work::default()
            )
            .unwrap(),
            Ok(())
        );
        settings.limits.max_file_bytes = "64 B".into();
        assert_eq!(
            verify_map_file(
                &store,
                root.path(),
                &settings,
                "large.rs",
                &[],
                64,
                Work::default()
            )
            .unwrap(),
            Err("source_changed")
        );
        settings.limits.max_file_bytes = "4 B".into();
        settings.chunk.target_tokens += 1;
        assert_eq!(
            verify_map_file(
                &store,
                root.path(),
                &settings,
                "large.rs",
                &[],
                4,
                Work::default()
            )
            .unwrap(),
            Err("l1_config_stale")
        );
        settings.chunk.target_tokens -= 1;
        std::fs::write(root.path().join("large.rs"), "larger source content now").unwrap();
        assert_eq!(
            verify_map_file(
                &store,
                root.path(),
                &settings,
                "large.rs",
                &[],
                4,
                Work::default()
            )
            .unwrap(),
            Err("source_changed")
        );
        assert_eq!(
            store
                .catalogue_source_checked("large.rs", &|| Ok(()))
                .unwrap()
                .unwrap(),
            before
        );
    }

    #[test]
    fn changed_age_handles_subseconds_future_and_unknown_timestamps() {
        let now = UNIX_EPOCH + Duration::from_secs(100);
        assert_eq!(changed_age(None, now), None);
        assert_eq!(changed_age(Some(now), now), Some(0));
        assert_eq!(
            changed_age(Some(now + Duration::from_secs(1)), now),
            Some(0)
        );
        assert_eq!(
            changed_age(Some(now - Duration::from_millis(1999)), now),
            Some(1)
        );
        assert_eq!(
            changed_age(Some(UNIX_EPOCH - Duration::from_secs(1)), now),
            Some(101)
        );
    }
}
