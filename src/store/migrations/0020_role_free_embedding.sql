-- v1 §7.1: prove role-only changes separately from source/model/policy repairs.
-- Legacy publications remain intact and unknown; their original desired key
-- can match a freshly recomputed role-free generation, never a guessed key.
ALTER TABLE embedding_publications ADD COLUMN role_free_hash BLOB
  CHECK(role_free_hash IS NULL OR (typeof(role_free_hash)='blob' AND length(role_free_hash)=32));
