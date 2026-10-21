-- v1 §7.3/§8: retained output identity is distinct from historical job completion.
-- Legacy rows have no proof; never infer one from a done job alone.
CREATE TABLE embedding_publications (
  unit_id INTEGER PRIMARY KEY REFERENCES units(id) ON DELETE CASCADE,
  table_slug TEXT NOT NULL REFERENCES embedding_tables(slug) ON DELETE CASCADE,
  desired_hash BLOB NOT NULL CHECK(length(desired_hash)=32)
);
