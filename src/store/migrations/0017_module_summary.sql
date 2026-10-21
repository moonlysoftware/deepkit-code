-- Module summary ownership and cleanup; v1 §6.8.
ALTER TABLE dirs ADD COLUMN summary_unit INTEGER REFERENCES units(id) ON DELETE SET NULL;
CREATE INDEX dirs_summary ON dirs(summary_unit) WHERE summary_unit IS NOT NULL;
