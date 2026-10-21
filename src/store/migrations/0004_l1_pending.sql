-- A catalogue write is not proof that its lexical/structural work completed.
-- Existing rows start pending so a normal index repairs pre-migration gaps.
ALTER TABLE files ADD COLUMN l1_pending INTEGER NOT NULL DEFAULT 1;
CREATE INDEX files_l1_pending ON files(l1_pending) WHERE l1_pending = 1;

CREATE TRIGGER files_l1_changed
AFTER UPDATE OF blake3, path, lang, kind, max_level, treatment ON files
WHEN NEW.blake3 IS NOT OLD.blake3 OR NEW.path IS NOT OLD.path
  OR NEW.lang IS NOT OLD.lang OR NEW.kind IS NOT OLD.kind
  OR NEW.max_level IS NOT OLD.max_level OR NEW.treatment IS NOT OLD.treatment
BEGIN
  UPDATE files SET l1_pending = 1 WHERE id = NEW.id;
END;
