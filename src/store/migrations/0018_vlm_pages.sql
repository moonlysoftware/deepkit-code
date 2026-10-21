-- v1 §6.2/§6.8/§8: reconstruct a File-target page job after a restart.
-- NULL preserves old/non-page jobs; page preparation refuses an absent ordinal.
ALTER TABLE jobs ADD COLUMN page INTEGER CHECK (
  page IS NULL OR (kind='enrich:vlm_page' AND target_kind='file'
                  AND page BETWEEN 1 AND 4294967295)
);

-- One complete generated page, not a document summary. The manifest detects
-- missing/partially removed sections; the caller re-proves current PDF input.
CREATE TABLE page_publications (
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  page INTEGER NOT NULL CHECK(page BETWEEN 1 AND 4294967295),
  parent_unit INTEGER NOT NULL UNIQUE REFERENCES units(id) ON DELETE CASCADE,
  desired_hash BLOB NOT NULL CHECK(length(desired_hash)=32),
  sections INTEGER NOT NULL CHECK(sections>0),
  output_hash BLOB NOT NULL CHECK(length(output_hash)=32),
  PRIMARY KEY(file_id,page)
);
