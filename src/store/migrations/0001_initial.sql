CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE files (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  blake3 BLOB NOT NULL,
  size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  lang TEXT,
  kind TEXT NOT NULL CHECK (kind IN ('code','doc','config','image','data','other')),
  git_blob BLOB,
  ladder_level INTEGER NOT NULL DEFAULT 0,
  max_level INTEGER NOT NULL,
  treatment TEXT NOT NULL,
  indexed_at INTEGER,
  needs_vlm INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE dirs (
  path TEXT PRIMARY KEY,
  merkle BLOB NOT NULL,
  file_count INTEGER NOT NULL,
  module_unit INTEGER REFERENCES units(id) ON DELETE SET NULL
);

CREATE TABLE units (
  id INTEGER PRIMARY KEY,
  hash BLOB NOT NULL UNIQUE,
  file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN
    ('chunk','symbol','doc_section','config_key','image','commit',
     'pr','history','module','summary')),
  byte_start INTEGER,
  byte_end INTEGER,
  line_start INTEGER,
  line_end INTEGER,
  scope_path TEXT,
  header TEXT NOT NULL,
  text TEXT NOT NULL,
  tokens INTEGER NOT NULL,
  ladder_level INTEGER NOT NULL DEFAULT 1,
  parent_unit INTEGER REFERENCES units(id) ON DELETE CASCADE,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX units_file ON units(file_id);
CREATE INDEX units_kind ON units(kind);

CREATE VIRTUAL TABLE fts_units USING fts5(
  header, idents, body, prose,
  content='', contentless_delete=1,
  tokenize="unicode61 remove_diacritics 2 tokenchars '_'"
);

CREATE VIRTUAL TABLE fts_names USING fts5(
  name, kind UNINDEXED, ref_id UNINDEXED,
  tokenize='trigram case_sensitive 0'
);

CREATE TABLE unit_sources (
  unit_id INTEGER NOT NULL REFERENCES units(id) ON DELETE CASCADE,
  source_kind TEXT NOT NULL CHECK (source_kind IN ('unit','file','commit','symbol')),
  source_id INTEGER NOT NULL,
  role TEXT,
  PRIMARY KEY (unit_id, source_kind, source_id)
);

CREATE TABLE symbols (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  unit_id INTEGER REFERENCES units(id) ON DELETE SET NULL,
  name TEXT NOT NULL,
  qualified TEXT,
  kind TEXT NOT NULL,
  signature TEXT,
  line_start INTEGER NOT NULL,
  line_end INTEGER NOT NULL,
  pagerank REAL NOT NULL DEFAULT 0,
  summary_unit INTEGER REFERENCES units(id) ON DELETE SET NULL,
  history_unit INTEGER REFERENCES units(id) ON DELETE SET NULL
);
CREATE INDEX symbols_name ON symbols(name);
CREATE INDEX symbols_file ON symbols(file_id);
CREATE INDEX symbols_rank ON symbols(pagerank DESC);

CREATE TABLE edges (
  src_symbol INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
  dst_name TEXT NOT NULL,
  dst_symbol INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
  kind TEXT NOT NULL CHECK (kind IN ('ref','import','doc_mention','test_of')),
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  line INTEGER
);
CREATE INDEX edges_dst_name ON edges(dst_name);
CREATE INDEX edges_dst_symbol ON edges(dst_symbol);

CREATE TABLE commits (
  id INTEGER PRIMARY KEY,
  sha BLOB NOT NULL UNIQUE,
  author TEXT,
  author_time INTEGER NOT NULL,
  subject TEXT NOT NULL,
  body TEXT,
  parents INTEGER NOT NULL,
  files_touched TEXT NOT NULL,
  symbols_touched TEXT,
  additions INTEGER,
  deletions INTEGER,
  pr_id INTEGER REFERENCES prs(id) ON DELETE SET NULL,
  unit_id INTEGER REFERENCES units(id) ON DELETE SET NULL
);
CREATE INDEX commits_time ON commits(author_time DESC);

CREATE TABLE prs (
  id INTEGER PRIMARY KEY,
  number INTEGER,
  title TEXT NOT NULL,
  merge_sha BLOB NOT NULL UNIQUE,
  base_sha BLOB,
  head_sha BLOB,
  strategy TEXT NOT NULL CHECK (strategy IN ('merge','squash','gitlab','plain')),
  commit_count INTEGER NOT NULL,
  merged_at INTEGER NOT NULL,
  unit_id INTEGER REFERENCES units(id) ON DELETE SET NULL,
  summary_unit INTEGER REFERENCES units(id) ON DELETE SET NULL
);

CREATE TABLE embedding_tables (
  slug TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  profile TEXT NOT NULL,
  dims INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE jobs (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  target_kind TEXT NOT NULL,
  target_id INTEGER NOT NULL,
  target_hash BLOB NOT NULL,
  provider TEXT NOT NULL,
  priority REAL NOT NULL,
  tokens INTEGER NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('queued','leased','done','failed','dead')),
  attempts INTEGER NOT NULL DEFAULT 0,
  not_before INTEGER NOT NULL DEFAULT 0,
  leased_until INTEGER,
  lease_owner TEXT,
  last_error TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE (kind, target_kind, target_id, target_hash)
);
CREATE INDEX jobs_claim ON jobs(state, provider, not_before, priority DESC);

CREATE TABLE usage (
  day TEXT NOT NULL,
  provider TEXT NOT NULL,
  calls INTEGER NOT NULL,
  tokens INTEGER NOT NULL,
  PRIMARY KEY (day, provider)
);

CREATE TABLE config_effective (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  source TEXT NOT NULL
);
