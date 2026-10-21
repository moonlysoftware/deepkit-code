-- v1 §7.1: last verified prepared context for per-table mismatch reporting.
-- Deliberately no legacy backfill: missing evidence remains unknown.
CREATE TABLE unit_contexts (
    unit_id INTEGER PRIMARY KEY REFERENCES units(id) ON DELETE CASCADE,
    content_hash BLOB NOT NULL CHECK(typeof(content_hash)='blob' AND length(content_hash)=32),
    ctx_hash BLOB NOT NULL CHECK(typeof(ctx_hash)='blob' AND length(ctx_hash)=32),
    recipe_hash BLOB NOT NULL CHECK(typeof(recipe_hash)='blob' AND length(recipe_hash)=32)
);
-- A retained occurrence can move or receive a new header without changing text.
-- Until preparation records the new header, status must report unknown.
CREATE TRIGGER units_invalidate_context AFTER UPDATE OF content_hash,header,scope_path,source_key,file_id,kind ON units
WHEN NEW.content_hash IS NOT OLD.content_hash OR NEW.header IS NOT OLD.header
  OR NEW.scope_path IS NOT OLD.scope_path OR NEW.source_key IS NOT OLD.source_key
  OR NEW.file_id IS NOT OLD.file_id OR NEW.kind IS NOT OLD.kind
BEGIN
    DELETE FROM unit_contexts WHERE unit_id=NEW.id;
END;
CREATE TRIGGER files_invalidate_context AFTER UPDATE OF path ON files
WHEN NEW.path IS NOT OLD.path
BEGIN
    DELETE FROM unit_contexts WHERE unit_id IN (
        SELECT id FROM units WHERE file_id=NEW.id
        UNION SELECT s.id FROM units s JOIN units p ON p.id=s.parent_unit WHERE p.file_id=NEW.id
        UNION SELECT history_unit FROM symbols WHERE file_id=NEW.id AND history_unit IS NOT NULL
    );
END;
