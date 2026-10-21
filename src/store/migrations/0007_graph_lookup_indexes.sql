-- Graph expansion probes symbols by owning unit and outgoing edges by source.
-- Without these indexes every lexical seed repeatedly scans the whole graph.
CREATE INDEX symbols_unit ON symbols(unit_id);
CREATE INDEX edges_src ON edges(src_symbol);
