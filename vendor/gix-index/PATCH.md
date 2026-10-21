# Temporary gix-index 0.55.0 patch

Authorized by accepted PCR-016. This is the existing transitive dependency,
not an additional Deepkit product crate. MIT OR Apache-2.0; licenses retained.
Published source commit: d2af4ed5532ea660fbd643e48d8925cd88de5ee0.
Original file hashes: notes/bench/t1-f1-vendor-provenance.json.

Only src/file/init.rs changes behavior: read owned bytes, check header/trailer
length before slicing, and hash the same snapshot passed to the decoder.
The normal split-index loader reuses this boundary for backing files. Existing
checksum semantics, including null trailers and skip_hash, remain unchanged.
No panic-catching, memory mapping or second hash read is used in this loader.

Regression tests live in src/sync/index_loader_tests.rs in the product crate.
Remove this directory and Cargo patch when a verified upstream release supplies
the same safety properties, then rerun the regression matrix and full gates.
No upstream message or accepted upstream fix is claimed.
