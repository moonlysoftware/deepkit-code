-- v1 §7.3: known provider identity for future profile-transition query routing.
-- Existing vectors deliberately remain unbound: their endpoint cannot be inferred.
CREATE TABLE embedding_bindings (
  slug TEXT PRIMARY KEY REFERENCES embedding_tables(slug) ON DELETE CASCADE,
  profile TEXT NOT NULL CHECK(profile IN ('local','remote','ollama')),
  model TEXT NOT NULL,
  dims INTEGER NOT NULL CHECK(dims > 0),
  endpoint TEXT NOT NULL,
  artifact_hash BLOB CHECK(artifact_hash IS NULL OR length(artifact_hash)=32),
  query_prefix TEXT NOT NULL,
  api_key_env TEXT
);
