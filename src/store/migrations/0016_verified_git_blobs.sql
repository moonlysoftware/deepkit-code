-- Earlier versions paired dirty working-tree digests with Git index OIDs.
-- Retain all indexed payloads, but require clean stat evidence before restoring
-- an OID association. Older binaries must not republish ambiguous associations.
UPDATE files SET git_blob=NULL WHERE git_blob IS NOT NULL;
