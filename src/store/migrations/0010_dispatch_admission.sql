-- v1 §8.2: retain dispatch instants for exact rolling request/token admission.
ALTER TABLE usage_reservations ADD COLUMN dispatched_at INTEGER;
-- Old accounting cannot prove dispatch time. Charge at upgrade time for one
-- conservative minute, preserving all daily totals and uncertain reservations.
UPDATE usage_reservations SET dispatched_at=CAST(strftime('%s','now') AS INTEGER)
WHERE state IN ('dispatched','reconciled');
CREATE INDEX usage_reservations_rate ON usage_reservations(provider,dispatched_at)
WHERE dispatched_at IS NOT NULL;
CREATE INDEX usage_reservations_daily ON usage_reservations(provider,day,state);
