-- v1 §7.1/§13: endpoint identity captured with potential dispatch, never guessed
-- from current configuration. Legacy reservations remain explicitly unknown.
ALTER TABLE usage_reservations ADD COLUMN endpoint_kind TEXT;
ALTER TABLE usage_reservations ADD COLUMN endpoint_url TEXT;
ALTER TABLE usage_reservations ADD COLUMN sent_units INTEGER CHECK(sent_units > 0);
