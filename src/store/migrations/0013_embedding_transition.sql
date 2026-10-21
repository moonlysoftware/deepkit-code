-- v1 §7.3: one serialized routing change, with bounded old sources until replaced.
CREATE TABLE embedding_transition (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  generation INTEGER NOT NULL CHECK(generation>0),
  desired TEXT NOT NULL CHECK(json_valid(desired)),
  previous TEXT NOT NULL CHECK(json_valid(previous))
);
