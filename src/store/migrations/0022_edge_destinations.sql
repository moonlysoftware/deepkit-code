-- v1 §6.6 / §7: resolve caller/doc and module graph inputs without scanning
-- every unresolved edge. The prefix also supports resolved symbol lookups.
CREATE INDEX edges_destination ON edges(dst_symbol,dst_name);
