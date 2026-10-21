-- v1 §3.2 / §4.3: per-file replacement must not scan unrelated graph/content.
-- SQLite does not create indexes for the child side of foreign keys. Nullable
-- links use partial indexes so an unused later-phase feature costs one page.
CREATE INDEX units_parent ON units(parent_unit) WHERE parent_unit IS NOT NULL;
CREATE INDEX dirs_module ON dirs(module_unit) WHERE module_unit IS NOT NULL;
CREATE INDEX symbols_summary ON symbols(summary_unit) WHERE summary_unit IS NOT NULL;
CREATE INDEX symbols_history ON symbols(history_unit) WHERE history_unit IS NOT NULL;
CREATE INDEX commits_unit ON commits(unit_id) WHERE unit_id IS NOT NULL;
CREATE INDEX prs_unit ON prs(unit_id) WHERE unit_id IS NOT NULL;
CREATE INDEX prs_summary ON prs(summary_unit) WHERE summary_unit IS NOT NULL;
CREATE INDEX edges_file ON edges(file_id);
CREATE INDEX unit_sources_source ON unit_sources(source_kind, source_id, unit_id);
-- The Rust migration rekeys existing fts_names rowids in this transaction.
