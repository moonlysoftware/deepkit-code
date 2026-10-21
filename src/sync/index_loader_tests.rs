//! Regression coverage for the PCR-016 Git-index boundary; v1 §4.6.
use std::{fs, path::Path, process::Command};

fn git(root: &Path, args: &[&str]) {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn fixture(format: &str) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    git(
        root.path(),
        &["init", "--quiet", &format!("--object-format={format}")],
    );
    fs::write(root.path().join("tracked.txt"), "index fixture\n").unwrap();
    git(root.path(), &["add", "tracked.txt"]);
    root
}

fn load(
    path: &Path,
    kind: gix::hash::Kind,
    skip_hash: bool,
) -> Result<gix::index::File, gix::index::file::init::Error> {
    gix::index::File::at(path, kind, skip_hash, Default::default())
}

#[test]
fn owned_git_index_refuses_sha1_sha256_truncation_and_checksum_corruption() {
    for (format, kind) in [
        ("sha1", gix::hash::Kind::Sha1),
        ("sha256", gix::hash::Kind::Sha256),
    ] {
        let root = fixture(format);
        let path = root.path().join(".git/index");
        let valid = fs::read(&path).unwrap();
        assert_eq!(load(&path, kind, false).unwrap().entries().len(), 1);
        for length in 0..12 + kind.len_in_bytes() {
            fs::write(&path, &valid[..length]).unwrap();
            for skip in [false, true] {
                assert!(
                    load(&path, kind, skip).is_err(),
                    "{format} length={length} skip={skip}"
                );
            }
        }
        fs::write(&path, b"invalid index").unwrap();
        assert!(load(&path, kind, false).is_err());
        let mut corrupt = valid.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 1;
        fs::write(&path, &corrupt).unwrap();
        assert!(load(&path, kind, false).is_err());
        fs::write(&path, valid).unwrap();
        assert_eq!(load(&path, kind, false).unwrap().entries().len(), 1);
    }
}

#[test]
fn owned_git_index_checks_split_backing_snapshots_in_both_hash_formats() {
    for (format, kind) in [
        ("sha1", gix::hash::Kind::Sha1),
        ("sha256", gix::hash::Kind::Sha256),
    ] {
        let root = fixture(format);
        git(root.path(), &["update-index", "--split-index"]);
        let path = root.path().join(".git/index");
        let backing = fs::read_dir(root.path().join(".git"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("sharedindex.")
            })
            .unwrap();
        let valid = fs::read(&backing).unwrap();
        assert_eq!(load(&path, kind, false).unwrap().entries().len(), 1);
        for length in [0, 1, 13, 12 + kind.len_in_bytes() - 1] {
            fs::write(&backing, &valid[..length]).unwrap();
            assert!(
                load(&path, kind, false).is_err(),
                "{format} backing length={length}"
            );
        }
        let mut corrupt = valid.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 1;
        fs::write(&backing, &corrupt).unwrap();
        assert!(load(&path, kind, false).is_err());
        fs::write(&backing, valid).unwrap();
        assert_eq!(load(&path, kind, false).unwrap().entries().len(), 1);
    }
}
