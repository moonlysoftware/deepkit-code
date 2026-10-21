ALTER TABLE files ADD COLUMN truncated INTEGER NOT NULL DEFAULT 0;
CREATE INDEX files_truncated ON files(truncated) WHERE truncated = 1;
