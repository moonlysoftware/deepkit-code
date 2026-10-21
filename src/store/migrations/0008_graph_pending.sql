-- Graph publication is durable work: a crash after L1 must not turn it into a no-op.
INSERT INTO meta(key, value) VALUES ('graph_pending', '1');
CREATE TRIGGER symbols_rank_pending_insert AFTER INSERT ON symbols BEGIN
  UPDATE meta SET value='1' WHERE key='graph_pending';
END;
CREATE TRIGGER symbols_rank_pending_delete AFTER DELETE ON symbols BEGIN
  UPDATE meta SET value='1' WHERE key='graph_pending';
END;
CREATE TRIGGER symbols_rank_pending_update AFTER UPDATE OF unit_id,file_id,name,qualified,kind ON symbols BEGIN
  UPDATE meta SET value='1' WHERE key='graph_pending';
END;
CREATE TRIGGER edges_rank_pending_insert AFTER INSERT ON edges BEGIN
  UPDATE meta SET value='1' WHERE key='graph_pending';
END;
CREATE TRIGGER edges_rank_pending_delete AFTER DELETE ON edges BEGIN
  UPDATE meta SET value='1' WHERE key='graph_pending';
END;
CREATE TRIGGER edges_rank_pending_update AFTER UPDATE ON edges BEGIN
  UPDATE meta SET value='1' WHERE key='graph_pending';
END;
