-- v1 §3.5/§8: an observed multi-section page must not retry into unchanged capacity.
ALTER TABLE jobs ADD COLUMN required_l3_units INTEGER NOT NULL DEFAULT 1 CHECK (
  required_l3_units BETWEEN 1 AND 4294967295
  AND (kind='enrich:vlm_page' OR required_l3_units=1)
);

-- Exact page ownership is queried during claims, publication and source proofs.
CREATE INDEX units_page_anchor ON units(file_id,duplicate_ordinal)
WHERE kind='image' AND ladder_level=0 AND parent_unit IS NULL;
