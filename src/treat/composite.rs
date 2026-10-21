//! Canonical composite payload identity and source fingerprints; v1.md §6.7.

fn field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Fingerprints the versioned source multiset, retaining duplicate contributions.
/// Length framing prevents variable-width kind, version and hash ambiguity.
pub(crate) fn source_fingerprint<'a>(
    kind: &str,
    version: &str,
    hashes: impl IntoIterator<Item = &'a [u8]>,
) -> [u8; 32] {
    let mut hashes = hashes.into_iter().collect::<Vec<_>>();
    hashes.sort_unstable();
    let mut hasher = blake3::Hasher::new();
    field(&mut hasher, kind.as_bytes());
    field(&mut hasher, version.as_bytes());
    for hash in hashes {
        field(&mut hasher, hash);
    }
    *hasher.finalize().as_bytes()
}

/// Identifies immutable collated text as well as its sources. Context, ownership,
/// models and policy generations deliberately remain outside this payload key.
pub(crate) fn payload_hash(fingerprint: &[u8; 32], text: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    field(&mut hasher, fingerprint);
    field(&mut hasher, text.as_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_tracks_source_multiset_version_and_body_without_input_order() {
        let first = source_fingerprint("module", "v1", [b"a".as_slice(), b"b"]);
        assert_eq!(
            first,
            source_fingerprint("module", "v1", [b"b".as_slice(), b"a"])
        );
        for other in [
            source_fingerprint("module", "v1", [b"a".as_slice()]),
            source_fingerprint("module", "v1", [b"a".as_slice(), b"a", b"b"]),
            source_fingerprint("module", "v2", [b"a".as_slice(), b"b"]),
            source_fingerprint("history", "v1", [b"a".as_slice(), b"b"]),
        ] {
            assert_ne!(first, other);
        }
        assert_ne!(
            payload_hash(&first, "files:\na.rs"),
            payload_hash(&first, "files:\nb.rs")
        );
        assert_ne!(
            source_fingerprint("a", "bc", []),
            source_fingerprint("ab", "c", [])
        );
        assert_ne!(
            source_fingerprint("module", "v1", [b"ab".as_slice(), b"c"]),
            source_fingerprint("module", "v1", [b"a".as_slice(), b"bc"])
        );
    }
}
