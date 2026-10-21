//! Shared per-file L1 preparation and publication; see v1.md §4.3 and §4.6.

use crate::{
    config::Config,
    store::{
        self, FtsColumns, GraphEdgeKind, IndexedUnitInput, Store, SymbolInsert, UnitEdgeInsert,
        UnitInsert,
    },
    sync::{Error, ScannedFile, unix_timestamp},
    treat::{
        PreparedUnit, Router,
        context::{ContextFile, ContextUnit, build, cap_units, unit_hash},
        fts::{FtsInput, build_columns},
        graph, prepare_units,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

/// Captured source and catalogue identity for one independently processed file.
#[derive(Clone, Copy)]
pub(crate) struct FileInput<'a> {
    pub(crate) work: super::Work<'a>,
    pub(crate) file: &'a ScannedFile,
    pub(crate) fence: store::l1::Fence<'a>,
}

/// Counts only admitted output, retaining visible truncation and cap deferrals.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct FileOutcome {
    pub(crate) units: usize,
    pub(crate) truncated: bool,
}

/// Owned preparation retains the exact input fence until serial publication.
pub(crate) struct PreparedFile<'a> {
    input: FileInput<'a>,
    output: PreparedOutput,
}

enum PreparedOutput {
    Empty,
    Deferred(store::l1::CapDeferral),
    Ready(ReadyFile),
}

struct ReadyFile {
    units: Vec<PreparedUnit>,
    graph: Option<graph::FileGraph>,
    symbol_groups: Vec<Vec<usize>>,
    computed: Vec<ComputedUnit>,
    completion: store::l1::Completion,
}

struct ComputedUnit {
    hash: [u8; 32],
    columns: crate::treat::fts::LexicalColumns,
    tokens: i64,
}

/// Reuses routing and repository context across a batch of file preparations.
pub(crate) struct Processor<'a> {
    root: &'a Path,
    settings: &'a Config,
    router: Router,
    repo_name: String,
    max_bytes: u64,
    source_limits: crate::sync::source_limits::SourceLimits,
}

impl<'a> Processor<'a> {
    /// Reconstructs exact lexical columns for retained parser occurrences in an
    /// existing transaction. No units, graphs or vectors are republished. Source
    /// bytes and L1 policy must still match the completed catalogue snapshot.
    pub(crate) fn lexical_file_in(
        &self,
        tx: &rusqlite::Transaction<'_>,
        file_id: i64,
        work: super::Work<'_>,
    ) -> Result<Vec<(i64, crate::treat::fts::LexicalColumns)>, Error> {
        use rusqlite::OptionalExtension;
        work.check()?;
        let record = tx.query_row(
            "SELECT path,blake3,size,mtime_ns,lang,kind,l1_fingerprint FROM files WHERE id=?1 AND l1_pending=0 AND l1_fingerprint=l1_desired",
            [file_id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?, r.get::<_, Option<String>>(4)?, r.get::<_, String>(5)?, r.get::<_, Vec<u8>>(6)?)),
        ).optional().map_err(store::Error::from)?;
        let Some((path, hash, size, mtime_ns, lang, kind, fingerprint)) = record else {
            return Err(store::Error::L1InputsChanged.into());
        };
        let file = ScannedFile {
            path,
            blake3: hash.try_into().map_err(|_| store::Error::L1InputsChanged)?,
            size: u64::try_from(size).map_err(|_| store::Error::L1InputsChanged)?,
            mtime_ns,
            lang,
            kind: store::FileKind::from_db(&kind).ok_or(store::Error::L1InputsChanged)?,
            content_capped: false,
            hash_reused_from_git: false,
            source_stamp: Default::default(),
        };
        let generation = tx
            .query_row(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'",
                [],
                |r| r.get(0),
            )
            .map_err(store::Error::from)?;
        let mut statement = tx.prepare("SELECT id,content_hash,duplicate_ordinal FROM units WHERE file_id=?1 AND parent_unit IS NULL").map_err(store::Error::from)?;
        let mut rows = statement.query([file_id]).map_err(store::Error::from)?;
        let mut identities = BTreeMap::new();
        while let Some(row) = rows.next().map_err(store::Error::from)? {
            work.check()?;
            let id: i64 = row.get(0).map_err(store::Error::from)?;
            let hash: Vec<u8> = row.get(1).map_err(store::Error::from)?;
            let ordinal: u32 = row.get(2).map_err(store::Error::from)?;
            identities.insert((hash, ordinal), id);
        }
        let prepared = self.prepare_file(
            FileInput {
                work,
                file: &file,
                fence: store::l1::Fence {
                    generation,
                    file_id,
                    path: &file.path,
                    content_hash: &file.blake3,
                    fingerprint: &fingerprint,
                },
            },
            identities.keys().cloned().collect(),
        )?;
        let PreparedOutput::Ready(ready) = prepared.output else {
            return Err(store::Error::L1InputsChanged.into());
        };
        let mut ordinals = BTreeMap::<[u8; 32], u32>::new();
        let mut output = Vec::new();
        for computed in ready.computed {
            work.check()?;
            let ordinal = ordinals.entry(computed.hash).or_default();
            let id = identities
                .get(&(computed.hash.to_vec(), *ordinal))
                .ok_or(store::Error::L1InputsChanged)?;
            *ordinal += 1;
            output.push((*id, computed.columns));
        }
        work.check()?;
        Ok(output)
    }

    /// Completes a targeted refresh only while the source remains discoverable.
    /// Recheck scope before preparation and inside each publication validation,
    /// including cap deferrals, because ignore rules can change after cataloguing.
    pub(crate) fn process_target(
        &self,
        store: &Store,
        input: FileInput<'_>,
        validate_settings: impl Fn() -> Result<(), store::Error>,
    ) -> Result<FileOutcome, Error> {
        let work = input.work;
        let path = &input.file.path;
        let validate_scope = || {
            work.check_store()?;
            if !crate::sync::target_eligible(self.root, path, work)
                .map_err(|_| store::Error::L1InputsChanged)?
            {
                return Err(store::Error::L1InputsChanged);
            }
            Ok(())
        };
        validate_scope()?;
        self.process_file(store, input, || {
            validate_settings()?;
            validate_scope()
        })
    }

    /// Captures effective preparation settings without reading or mutating sources.
    pub(crate) fn new(root: &'a Path, settings: &'a Config) -> Result<Self, Error> {
        Ok(Self {
            root,
            settings,
            router: Router::new(&settings.treatments, settings.llm.enabled)?,
            repo_name: super::repo_name(root, settings),
            max_bytes: settings.limits.max_file_bytes()?.0,
            source_limits: crate::sync::source_limits::SourceLimits::new(&settings.limits)?,
        })
    }

    /// Prepares contextual units, lexical columns and graph rows outside the writer,
    /// then publishes or defers atomically. The caller revalidates effective settings
    /// at publication; source validation and catalogue fences are always applied here.
    /// Cooperative work checks fence preparation and publication; individual calls
    /// are not preempted while they run.
    pub(crate) fn process_file(
        &self,
        store: &Store,
        input: FileInput<'_>,
        validate_settings: impl Fn() -> Result<(), store::Error>,
    ) -> Result<FileOutcome, Error> {
        input.work.check()?;
        // Reject mixed identities before using the supplied fence for any read.
        if input.fence.path != input.file.path || input.fence.content_hash != input.file.blake3 {
            return Err(store::Error::L1InputsChanged.into());
        }
        let existing =
            store.file_occurrence_keys_checked(input.fence, &|| input.work.check_store())?;
        let prepared = self.prepare_file(input, existing)?;
        self.publish_prepared(store, prepared, validate_settings)
    }

    /// Computes owned L1 output without SQLite access. A bounded set of files may
    /// prepare concurrently; the original work budget and source proof travel with it.
    pub(crate) fn prepare_file<'b>(
        &self,
        input: FileInput<'b>,
        existing: BTreeSet<store::l1::OccurrenceKey>,
    ) -> Result<PreparedFile<'b>, Error> {
        let FileInput { file, fence, work } = input;
        work.check()?;
        let Self {
            root,
            settings,
            router,
            repo_name,
            max_bytes,
            source_limits,
        } = self;
        let max_bytes = *max_bytes;
        let path = file.path.as_str();
        if fence.path != path || fence.content_hash != file.blake3 {
            return Err(store::Error::L1InputsChanged.into());
        }
        let treatment = router.route(path, file.kind);
        let fingerprint = super::file_fingerprint(
            settings,
            repo_name,
            super::FingerprintFile {
                path,
                content_hash: &file.blake3,
                language: file.lang.as_deref(),
                kind: file.kind,
                capped: file.content_capped,
            },
            &treatment,
        );
        if fence.fingerprint != fingerprint {
            return Err(store::Error::L1InputsChanged.into());
        }
        if treatment.max_level < 1 || file.content_capped {
            let output = if file.content_capped && !existing.is_empty() {
                PreparedOutput::Deferred(store::l1::CapDeferral::SourceBytes)
            } else {
                PreparedOutput::Empty
            };
            return Ok(PreparedFile { input, output });
        }
        if let Some(reason) = source_limits.refusal(file.kind, file.size) {
            return Ok(PreparedFile {
                input,
                output: PreparedOutput::Deferred(reason),
            });
        }
        let source = super::read_source(root, file, max_bytes, work)?;
        let mut completion = store::l1::Completion::default();
        let prepared = if treatment.chunker == crate::treat::Chunker::Pdf {
            let extracted = crate::treat::pdf::extract(
                &source,
                source_limits.pdf_pages(u64::MAX).admitted,
                &settings.chunk,
                settings.limits.max_units_per_file,
                || work.check(),
            )?;
            completion = store::l1::Completion {
                truncated: extracted.truncated,
                needs_vlm: extracted.needs_vlm,
                reason: extracted.reason,
            };
            extracted.units
        } else if treatment.chunker == crate::treat::Chunker::Notebook {
            let extracted = crate::treat::notebook::extract(
                &source,
                &settings.chunk,
                settings.limits.max_units_per_file,
                || work.check(),
            )?;
            completion.reason = extracted.reason;
            extracted.units
        } else if treatment.chunker == crate::treat::Chunker::Office {
            let extracted = crate::treat::office::extract(
                path,
                &source,
                &settings.chunk,
                settings.limits.max_units_per_file,
                || work.check(),
            )?;
            completion.reason = extracted.reason;
            extracted.units
        } else {
            prepare_units(
                path,
                file.lang.as_deref(),
                treatment.chunker,
                &source,
                &settings.chunk,
                settings.limits.max_units_per_file,
            )?
        };
        work.check()?;
        let mut ordinals = BTreeMap::<[u8; 32], u32>::new();
        let retained = prepared
            .iter()
            .take(usize::try_from(settings.limits.max_units_per_file).unwrap_or(usize::MAX))
            .map(|unit| {
                work.check()?;
                let hash = unit_hash(
                    unit.kind.as_str(),
                    (!unit.scope_path.is_empty()).then_some(unit.scope_path.as_str()),
                    &unit.text,
                );
                let ordinal = ordinals.entry(hash).or_default();
                let key = (hash.to_vec(), *ordinal);
                *ordinal += 1;
                Ok(key)
            })
            .collect::<Result<BTreeSet<_>, Error>>()?;
        let would_evict = (prepared.len() as u64 > settings.limits.max_units_per_file
            || completion.truncated)
            && existing.iter().any(|key| !retained.contains(key));
        if would_evict {
            return Ok(PreparedFile {
                input,
                output: PreparedOutput::Deferred(if completion.reason == Some("max_pdf_pages") {
                    store::l1::CapDeferral::PdfPages
                } else {
                    store::l1::CapDeferral::Units
                }),
            });
        }
        let capped = cap_units(prepared, settings.limits.max_units_per_file);
        completion.truncated |= capped.truncated || capped.units.iter().any(|unit| unit.truncated);
        let prepared_units = capped.units;
        work.check()?;
        let graph = file
            .lang
            .as_deref()
            .and_then(|language| graph::extract_file_graph(language, path, &source).ok())
            .flatten();
        work.check()?;
        let symbol_groups = graph
            .as_ref()
            .map(|graph| graph_symbol_groups(graph, &prepared_units, work))
            .transpose()?
            .unwrap_or_else(|| vec![Vec::new(); prepared_units.len()]);
        let mut computed = Vec::with_capacity(prepared_units.len());
        for prepared in &prepared_units {
            work.check()?;
            let idents = prepared
                .idents
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            let context = build(
                ContextUnit {
                    scope_path: (!prepared.scope_path.is_empty()).then_some(&prepared.scope_path),
                    header: &prepared.header,
                    language: file.lang.as_deref(),
                    kind: embedding_kind(&prepared.kind),
                    text: &prepared.text,
                    is_document: prepared.kind == store::UnitKind::DocSection,
                },
                ContextFile { repo_name, path },
                &[],
            );
            let (lexical_kind, lexical_language) = match &prepared.origin {
                crate::treat::TextOrigin::File => (file.kind, file.lang.as_deref()),
                crate::treat::TextOrigin::EmbeddedCode { language } => {
                    (store::FileKind::Code, language.as_deref())
                }
            };
            let columns = build_columns(
                FtsInput {
                    breadcrumb: &context.breadcrumb,
                    header: &context.header,
                    text: &context.text,
                    language: lexical_language,
                    kind: lexical_kind,
                    extra_idents: &idents,
                    has_code_fence: prepared.has_code_fence,
                },
                &settings.tokenize,
            );
            let hash = unit_hash(
                prepared.kind.as_str(),
                (!prepared.scope_path.is_empty()).then_some(prepared.scope_path.as_str()),
                &prepared.text,
            );
            computed.push(ComputedUnit {
                hash,
                columns,
                tokens: i64::try_from(token_count(&prepared.text)).unwrap_or(i64::MAX),
            });
        }
        work.check()?;
        Ok(PreparedFile {
            input,
            output: PreparedOutput::Ready(ReadyFile {
                units: prepared_units,
                graph,
                symbol_groups,
                computed,
                completion,
            }),
        })
    }

    /// Publishes prepared output in caller order, rechecking source, configuration
    /// and catalogue identity under the writer lock before committing any result.
    pub(crate) fn publish_prepared(
        &self,
        store: &Store,
        prepared: PreparedFile<'_>,
        validate_settings: impl Fn() -> Result<(), store::Error>,
    ) -> Result<FileOutcome, Error> {
        let PreparedFile {
            input: FileInput { file, fence, work },
            output,
        } = prepared;
        work.check()?;
        let check = || work.check_store();
        let admission_check = work.admission_check(&check);
        let validate_inputs = || {
            work.check_store()?;
            validate_settings()?;
            work.check_store()?;
            super::verify_source(self.root, file, self.max_bytes, work)
                .map_err(|_| store::Error::L1InputsChanged)
        };
        let ready = match output {
            PreparedOutput::Deferred(reason) => {
                store.defer_file_l1(admission_check, fence, reason, validate_inputs)?;
                return Ok(FileOutcome {
                    units: 0,
                    truncated: true,
                });
            }
            PreparedOutput::Empty => {
                let outcome = store.publish_file_content(
                    admission_check,
                    fence,
                    &[],
                    false,
                    validate_inputs,
                )?;
                return Ok(FileOutcome {
                    units: 0,
                    truncated: outcome == store::PublicationOutcome::StorageDeferred,
                });
            }
            PreparedOutput::Ready(ready) => ready,
        };
        let ReadyFile {
            units: prepared_units,
            graph,
            symbol_groups,
            computed,
            completion,
        } = ready;
        let mut ordinals = BTreeMap::<[u8; 32], u32>::new();
        let mut inputs = Vec::with_capacity(prepared_units.len());
        for (unit_index, (prepared, computed)) in prepared_units.iter().zip(&computed).enumerate() {
            work.check()?;
            let hash = &computed.hash;
            let columns = &computed.columns;
            let ordinal = ordinals.entry(*hash).or_default();
            let duplicate_ordinal = *ordinal;
            *ordinal += 1;
            let symbol_indices = &symbol_groups[unit_index];
            let symbols = graph.as_ref().map_or_else(Vec::new, |graph| {
                symbol_indices
                    .iter()
                    .map(|index| {
                        let symbol = &graph.symbols[*index];
                        SymbolInsert {
                            unit_id: None,
                            name: &symbol.name,
                            qualified: (!symbol.qualified.is_empty()).then_some(&symbol.qualified),
                            kind: &symbol.kind,
                            signature: (!symbol.signature.is_empty()).then_some(&symbol.signature),
                            line_start: i64::try_from(symbol.line_start).unwrap_or(i64::MAX),
                            line_end: i64::try_from(symbol.line_end).unwrap_or(i64::MAX),
                        }
                    })
                    .collect::<Vec<_>>()
            });
            let edges = graph.as_ref().map_or_else(Vec::new, |graph| {
                graph
                    .edges
                    .iter()
                    .filter_map(|edge| {
                        symbol_indices
                            .iter()
                            .position(|index| *index == edge.source_index)
                            .map(|src_symbol_index| UnitEdgeInsert {
                                src_symbol_index,
                                dst_name: &edge.dst_name,
                                dst_symbol: None,
                                dst_symbol_index: None,
                                kind: match edge.kind {
                                    graph::EdgeKind::Ref => GraphEdgeKind::Ref,
                                    graph::EdgeKind::Import => GraphEdgeKind::Import,
                                },
                                line: i64::try_from(edge.line).ok(),
                            })
                    })
                    .collect::<Vec<_>>()
            });
            inputs.push(IndexedUnitInput {
                unit: UnitInsert {
                    hash,
                    source_key: None,
                    duplicate_ordinal,
                    file_id: Some(fence.file_id),
                    kind: prepared.kind,
                    byte_start: None,
                    byte_end: None,
                    line_start: i64::try_from(prepared.line_start).ok(),
                    line_end: i64::try_from(prepared.line_end).ok(),
                    scope_path: (!prepared.scope_path.is_empty()).then_some(&prepared.scope_path),
                    header: &prepared.header,
                    text: &prepared.text,
                    tokens: computed.tokens,
                    ladder_level: 1,
                    parent_unit: None,
                    created_at: unix_timestamp(),
                    updated_at: unix_timestamp(),
                    fts: FtsColumns {
                        header: &columns.header,
                        idents: &columns.idents,
                        body: &columns.body,
                        prose: &columns.prose,
                    },
                },
                symbols,
                edges,
            });
        }
        work.check()?;
        let outcome = store.publish_file_content(
            admission_check,
            fence,
            &inputs,
            completion,
            validate_inputs,
        )?;
        Ok(FileOutcome {
            units: if outcome == store::PublicationOutcome::Indexed {
                inputs.len()
            } else {
                0
            },
            truncated: completion.truncated
                || outcome == store::PublicationOutcome::StorageDeferred,
        })
    }
}

fn embedding_kind(kind: &store::UnitKind) -> &'static str {
    match kind {
        store::UnitKind::Chunk => "chunk",
        store::UnitKind::Symbol => "symbol",
        store::UnitKind::DocSection => "section",
        store::UnitKind::ConfigKey => "key",
        store::UnitKind::Image => "image",
        store::UnitKind::Commit => "commit",
        store::UnitKind::PullRequest => "pr",
        store::UnitKind::History => "history",
        store::UnitKind::Module => "module",
        store::UnitKind::Summary => "summary",
    }
}

fn graph_symbol_groups(
    graph: &graph::FileGraph,
    units: &[PreparedUnit],
    work: super::Work<'_>,
) -> Result<Vec<Vec<usize>>, Error> {
    let mut groups = vec![Vec::new(); units.len()];
    for (symbol_index, symbol) in graph.symbols.iter().enumerate() {
        work.check()?;
        let owner = units
            .iter()
            .position(|unit| {
                unit.line_start <= symbol.line_start && symbol.line_start <= unit.line_end
            })
            .or((!units.is_empty()).then_some(0));
        if let Some(owner) = owner {
            groups[owner].push(symbol_index);
        }
    }
    Ok(groups)
}

fn token_count(text: &str) -> usize {
    crate::treat::count_tokens(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::{self, SyncLimits, l1};
    use std::fs;

    #[test]
    fn shared_processor_bounds_occurrence_reader_admission_and_retries() {
        use std::{
            sync::atomic::{AtomicBool, Ordering},
            time::{Duration, Instant},
        };
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("source.rs"), "pub fn source() {}\n").unwrap();
        let store = store::open(":memory:", Default::default()).unwrap();
        let settings = Config::default();
        let scan = sync::scan(root.path(), SyncLimits::default()).unwrap();
        let publication = l1::publish(
            &store,
            l1::PublishInput {
                work: Default::default(),
                scan: &scan,
                git: None,
                snapshot: &store.index_snapshot().unwrap(),
                generation: 0,
                root: root.path(),
                settings: &settings,
                full: false,
            },
        )
        .unwrap();
        let file = scan.file("source.rs").unwrap();
        let snapshot = store.index_snapshot().unwrap();
        let record = &snapshot.files[0];
        let fence = store::l1::Fence {
            generation: publication.generation,
            file_id: record.id,
            path: &file.path,
            content_hash: &record.blake3,
            fingerprint: &publication.fingerprints[&file.path],
        };
        let processor = Processor::new(root.path(), &settings).unwrap();
        for cancel in [false, true] {
            let held = store.writer().unwrap();
            let cancelled = AtomicBool::new(false);
            let started = Instant::now();
            std::thread::scope(|scope| {
                if cancel {
                    scope.spawn(|| {
                        std::thread::sleep(Duration::from_millis(20));
                        cancelled.store(true, Ordering::Release);
                    });
                }
                let result = processor.process_file(
                    &store,
                    FileInput {
                        file,
                        fence,
                        work: l1::Work {
                            deadline: Some(
                                started + Duration::from_millis(if cancel { 500 } else { 20 }),
                            ),
                            cancelled: Some(&cancelled),
                        },
                    },
                    || panic!("contended reader must refuse before publication validation"),
                );
                assert!(matches!(
                    result,
                    Err(Error::Store(store::Error::L1InputsChanged))
                ));
            });
            assert!(started.elapsed() < Duration::from_secs(1));
            drop(held);
            assert!(store.file_occurrence_keys(record.id).unwrap().is_empty());
            assert_eq!(store.pending_l1_paths().unwrap(), ["source.rs"]);
            assert_eq!(store.l1_generation().unwrap(), publication.generation);
        }
        let outcome = processor
            .process_file(
                &store,
                FileInput {
                    file,
                    fence,
                    work: Default::default(),
                },
                || Ok(()),
            )
            .unwrap();
        assert!(outcome.units > 0);
        assert!(!outcome.truncated);
        assert!(store.pending_l1_paths().unwrap().is_empty());
    }

    #[test]
    fn shared_file_processing_fences_inputs_and_rolls_back_failed_revalidation() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        let source = root.path().join("source.rs");
        fs::write(&source, "pub fn original() {}\n").unwrap();
        let settings = Config::default();
        let store = store::open(root.path().join(".deepkit/code.db"), Default::default()).unwrap();
        let mut previous = BTreeSet::new();
        for replacement in [false, true] {
            if replacement {
                fs::write(&source, "pub fn replacement() {}\n").unwrap();
            }
            let scan = sync::scan(root.path(), SyncLimits::default()).unwrap();
            let publication = l1::publish(
                &store,
                l1::PublishInput {
                    work: Default::default(),
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
            let file = scan.file("source.rs").unwrap();
            let snapshot = store.index_snapshot().unwrap();
            let record = snapshot.files.iter().find(|f| f.path == file.path).unwrap();
            let fence = store::l1::Fence {
                generation: publication.generation,
                file_id: record.id,
                path: &file.path,
                content_hash: &record.blake3,
                fingerprint: &publication.fingerprints[&file.path],
            };
            let input = || FileInput {
                file,
                fence,
                work: Default::default(),
            };
            let processor = Processor::new(root.path(), &settings).unwrap();
            // The extracted API must reject mixed source/configuration identities
            // before a caller can publish output under a different fingerprint.
            let mut changed = settings.clone();
            changed.index.repo_name = "different-context".into();
            assert!(matches!(
                Processor::new(root.path(), &changed).unwrap().process_file(
                    &store,
                    input(),
                    || Ok(())
                ),
                Err(Error::Store(store::Error::L1InputsChanged))
            ));
            for invalid in [
                store::l1::Fence {
                    path: "other.rs",
                    ..fence
                },
                store::l1::Fence {
                    content_hash: &[0; 32],
                    ..fence
                },
            ] {
                assert!(matches!(
                    processor.process_file(
                        &store,
                        FileInput {
                            fence: invalid,
                            ..input()
                        },
                        || Ok(())
                    ),
                    Err(Error::Store(store::Error::L1InputsChanged))
                ));
            }
            // Real checked admission must stop at a competing SQLite writer,
            // before expensive validation or any replacement, for both generations.
            {
                use std::{
                    cell::Cell,
                    time::{Duration, Instant},
                };
                let other =
                    store::open(root.path().join(".deepkit/code.db"), Default::default()).unwrap();
                let mut writer = other.writer().unwrap();
                let held = writer
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                    .unwrap();
                let validations = Cell::new(0);
                let started = Instant::now();
                let result = processor.process_file(
                    &store,
                    FileInput {
                        work: l1::Work {
                            deadline: Some(started + Duration::from_millis(100)),
                            cancelled: None,
                        },
                        ..input()
                    },
                    || {
                        validations.set(validations.get() + 1);
                        Ok(())
                    },
                );
                assert!(matches!(
                    result,
                    Err(Error::Store(store::Error::L1InputsChanged)) | Err(Error::WorkStopped)
                ));
                assert!(started.elapsed() < Duration::from_secs(2));
                assert_eq!(validations.get(), 0);
                drop(held);
                assert_eq!(store.file_occurrence_keys(record.id).unwrap(), previous);
                assert!(store.pending_l1_paths().unwrap().contains(&file.path));
            }
            // Refusal after parsing and SQL replacement must preserve both the
            // old occurrences and pending work, including a previously indexed file.
            assert!(matches!(
                processor.process_file(&store, input(), || Err(store::Error::L1InputsChanged)),
                Err(Error::Store(store::Error::L1InputsChanged))
            ));
            assert_eq!(store.file_occurrence_keys(record.id).unwrap(), previous);
            assert!(store.pending_l1_paths().unwrap().contains(&file.path));
            #[cfg(unix)]
            {
                // The final in-transaction validation must reject even identical
                // bytes redirected outside the repository after preparation.
                let outside = tempfile::tempdir().unwrap();
                let external = outside.path().join("source.rs");
                fs::copy(&source, &external).unwrap();
                let validations = std::cell::Cell::new(0);
                assert!(matches!(
                    processor.process_file(&store, input(), || {
                        validations.set(validations.get() + 1);
                        if validations.get() == 2 {
                            fs::remove_file(&source).unwrap();
                            std::os::unix::fs::symlink(&external, &source).unwrap();
                        }
                        Ok(())
                    }),
                    Err(Error::Store(store::Error::L1InputsChanged))
                ));
                assert_eq!(store.file_occurrence_keys(record.id).unwrap(), previous);
                assert!(store.pending_l1_paths().unwrap().contains(&file.path));
                fs::remove_file(&source).unwrap();
                fs::copy(&external, &source).unwrap();
            }
            // A timeout before processing does no publication work. Cancellation
            // during the final settings check rolls back the replacement as well.
            use std::sync::atomic::{AtomicBool, Ordering};
            let cancelled = AtomicBool::new(false);
            let cancelled_input = || FileInput {
                work: super::super::Work {
                    deadline: None,
                    cancelled: Some(&cancelled),
                },
                ..input()
            };
            let validations = std::cell::Cell::new(0);
            assert!(matches!(
                processor.process_file(&store, cancelled_input(), || {
                    validations.set(validations.get() + 1);
                    if validations.get() == 2 {
                        cancelled.store(true, Ordering::Release);
                    }
                    Ok(())
                }),
                Err(Error::Store(store::Error::L1InputsChanged))
            ));
            assert_eq!(validations.get(), 2);
            assert_eq!(store.file_occurrence_keys(record.id).unwrap(), previous);
            assert!(store.pending_l1_paths().unwrap().contains(&file.path));
            assert!(matches!(
                processor.process_file(&store, cancelled_input(), || {
                    panic!("cancelled work must not reach validation")
                }),
                Err(Error::WorkStopped)
            ));
            assert!(matches!(
                processor.process_file(
                    &store,
                    FileInput {
                        work: super::super::Work {
                            deadline: Some(std::time::Instant::now()),
                            cancelled: None
                        },
                        ..input()
                    },
                    || panic!("expired work must not reach validation")
                ),
                Err(Error::WorkStopped)
            ));
            let result = processor.process_file(&store, input(), || Ok(())).unwrap();
            assert!(result.units > 0);
            assert!(!result.truncated);
            assert!(!store.pending_l1_paths().unwrap().contains(&file.path));
            let symbols = store.tool_symbols().unwrap();
            assert!(symbols.iter().any(|s| s.name
                == if replacement {
                    "replacement"
                } else {
                    "original"
                }));
            assert!(!replacement || symbols.iter().all(|s| s.name != "original"));
            previous = store.file_occurrence_keys(record.id).unwrap();
        }
    }
}
