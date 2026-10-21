-- v1 §8: full desired identity and renewable, fenced leases.
ALTER TABLE jobs RENAME TO jobs_legacy;
DROP INDEX jobs_claim;
CREATE TABLE jobs (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  target_kind TEXT NOT NULL,
  target_id INTEGER NOT NULL,
  target_hash BLOB NOT NULL,
  desired_hash BLOB NOT NULL,
  provider TEXT NOT NULL,
  priority REAL NOT NULL,
  tokens INTEGER NOT NULL CHECK(tokens >= 0),
  state TEXT NOT NULL CHECK(state IN ('queued','leased','done','failed','dead')),
  attempts INTEGER NOT NULL DEFAULT 0,
  not_before INTEGER NOT NULL DEFAULT 0,
  leased_until INTEGER,
  lease_owner TEXT,
  lease_generation INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE(kind, target_kind, target_id, desired_hash)
);
-- Old rows cannot supply a full desired fingerprint. Retain them for diagnosis,
-- with a reserved non-digest identity, but never dispatch pending legacy work.
INSERT INTO jobs
SELECT id, kind, target_kind, target_id, target_hash,
       CAST('legacy-job:' || id AS BLOB), provider, priority, tokens,
       CASE WHEN state IN ('done','dead') THEN state ELSE 'dead' END,
       attempts, not_before, NULL, NULL, 0,
       COALESCE(last_error || '; ', '') || 'legacy job lacks desired fingerprint; re-enqueue required',
       created_at, updated_at
FROM jobs_legacy;
DROP TABLE jobs_legacy;
CREATE INDEX jobs_claim ON jobs(state, provider, not_before, priority DESC);
ALTER TABLE usage ADD COLUMN estimated_tokens INTEGER NOT NULL DEFAULT 0;
CREATE TABLE provider_admission (
  provider TEXT PRIMARY KEY, request_window INTEGER NOT NULL, requests INTEGER NOT NULL,
  token_window INTEGER NOT NULL, tokens INTEGER NOT NULL
);
CREATE TABLE usage_reservations (
  id TEXT PRIMARY KEY, provider TEXT NOT NULL, day TEXT NOT NULL,
  job_id INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
  lease_generation INTEGER, calls INTEGER NOT NULL, tokens INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('reserved','dispatched','reconciled','released')),
  created_at INTEGER NOT NULL
);
