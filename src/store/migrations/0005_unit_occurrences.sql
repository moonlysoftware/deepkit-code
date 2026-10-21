-- Executed with foreign keys disabled outside the enclosing transaction, then
-- checked before commit. Preserve rowids referenced by vectors and graph rows.
CREATE TABLE unit_contents (
  hash BLOB NOT NULL PRIMARY KEY,
  text TEXT NOT NULL,
  tokens INTEGER NOT NULL
);
INSERT INTO unit_contents SELECT hash, text, tokens FROM units;

CREATE TABLE units_new (
  id INTEGER PRIMARY KEY,
  hash BLOB NOT NULL UNIQUE,
  content_hash BLOB NOT NULL REFERENCES unit_contents(hash),
  source_key TEXT NOT NULL,
  duplicate_ordinal INTEGER NOT NULL DEFAULT 0 CHECK (duplicate_ordinal >= 0),
  file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN
    ('chunk','symbol','doc_section','config_key','image','commit',
     'pr','history','module','summary')),
  byte_start INTEGER, byte_end INTEGER,
  line_start INTEGER, line_end INTEGER,
  scope_path TEXT,
  header TEXT NOT NULL,
  ladder_level INTEGER NOT NULL DEFAULT 1,
  parent_unit INTEGER REFERENCES units(id) ON DELETE CASCADE,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
INSERT INTO units_new
SELECT u.id, u.hash, u.hash,
       COALESCE(f.path, u.kind || ':' || lower(hex(u.hash))), 0,
       u.file_id, u.kind, u.byte_start, u.byte_end, u.line_start, u.line_end,
       u.scope_path, u.header, u.ladder_level, u.parent_unit, u.created_at, u.updated_at
FROM units u LEFT JOIN files f ON f.id = u.file_id;
DROP TABLE units;
ALTER TABLE units_new RENAME TO units;
CREATE INDEX units_file ON units(file_id);
CREATE INDEX units_content ON units(content_hash);
CREATE INDEX units_kind ON units(kind);

-- All sources must be reparsed; shared file provenance never contained the
-- other occurrence's ranges or contextual embedding input.
UPDATE files SET l1_pending = 1;
UPDATE symbols SET unit_id = NULL
-- Probe the referenced row once per symbol; a correlated IN over every other
-- file materializes a near-full unit set for every symbol on large upgrades.
WHERE EXISTS (
  SELECT 1 FROM units
  WHERE id = symbols.unit_id AND file_id != symbols.file_id
);

CREATE TRIGGER units_remove_derived AFTER DELETE ON units
BEGIN
  DELETE FROM fts_units WHERE rowid = OLD.id;
  DELETE FROM unit_contents WHERE hash = OLD.content_hash
    AND NOT EXISTS (SELECT 1 FROM units WHERE content_hash = OLD.content_hash);
END;
