-- Completion records exactly which source/context/configuration was published.
ALTER TABLE files ADD COLUMN l1_desired BLOB;
ALTER TABLE files ADD COLUMN l1_fingerprint BLOB;
ALTER TABLE files ADD COLUMN l1_deferred_reason TEXT;
UPDATE files SET l1_pending = 1;
INSERT INTO meta(key, value) VALUES ('l1_scan_generation', '0');
INSERT INTO meta(key, value) VALUES ('l1_completed_generation', '0');
-- Earlier versions advanced HEAD at catalogue publication; that is unproven.
UPDATE meta SET value = '' WHERE key = 'indexed_head';
