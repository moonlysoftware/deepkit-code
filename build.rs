//! Embed compile-time provenance so benchmarks do not identify old binaries by today's HEAD.
use std::{
    env, fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::Command,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let mut files = Vec::new();
    for name in [
        "src",
        "crates",
        "vendor",
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        ".cargo/config.toml",
    ] {
        println!("cargo:rerun-if-changed={name}");
        collect(&root.join(name), &mut files)?;
    }
    files.sort();
    let mut hash = blake3::Hasher::new();
    hash.update(b"dkc-build-inputs-v1\0");
    for path in files {
        let relative = path.strip_prefix(&root)?.to_string_lossy();
        hash.update(&(relative.len() as u64).to_le_bytes());
        hash.update(relative.as_bytes());
        hash.update(&fs::metadata(&path)?.len().to_le_bytes());
        let mut file = fs::File::open(path)?;
        let mut buffer = [0; 65536];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
    }
    emit("SOURCE_BLAKE3", hash.finalize().to_hex().as_ref());
    // Watching the resolved ref supports ordinary checkouts and linked worktrees.
    for name in ["HEAD", "index"] {
        if let Some(path) = git(&root, &["rev-parse", "--git-path", name]) {
            println!("cargo:rerun-if-changed={}", root.join(path).display());
        }
    }
    if let Some(reference) = git(&root, &["symbolic-ref", "-q", "HEAD"])
        && let Some(path) = git(&root, &["rev-parse", "--git-path", &reference])
    {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    emit(
        "COMMIT",
        &git(&root, &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned()),
    );
    let dirty = git(
        &root,
        &[
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "crates",
            "vendor",
            "Cargo.toml",
            "Cargo.lock",
            "build.rs",
            ".cargo",
        ],
    )
    .map(|status| if status.is_empty() { "false" } else { "true" })
    .unwrap_or("unknown");
    emit("SOURCE_DIRTY", dirty);
    for key in ["PROFILE", "OPT_LEVEL", "TARGET", "HOST"] {
        emit(key, &env::var(key)?);
    }
    let features = env::vars()
        .filter_map(|(name, _)| name.strip_prefix("CARGO_FEATURE_").map(str::to_owned))
        .collect::<std::collections::BTreeSet<_>>();
    emit(
        "FEATURES",
        &features.into_iter().collect::<Vec<_>>().join(","),
    );
    let mut flags = blake3::Hasher::new();
    for key in [
        "CARGO_ENCODED_RUSTFLAGS",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "CC",
        "CXX",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        flags.update(key.as_bytes());
        flags.update(&[0]);
        flags.update(env::var(key).unwrap_or_default().as_bytes());
        flags.update(&[0]);
    }
    emit("FLAGS_BLAKE3", flags.finalize().to_hex().as_ref());
    let rustc = Command::new(env::var("RUSTC")?).arg("--version").output()?;
    emit("RUSTC", String::from_utf8_lossy(&rustc.stdout).trim());
    Ok(())
}

fn emit(key: &str, value: &str) {
    println!("cargo:rustc-env=DKC_BUILD_{key}={value}");
}
fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
fn collect(path: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            if [".git", "target"]
                .iter()
                .any(|name| entry.file_name() == *name)
            {
                continue;
            }
            if entry.file_type()?.is_symlink() && entry.path().is_dir() {
                continue;
            }
            collect(&entry.path(), files)?;
        }
    } else {
        files.push(path.to_owned());
    }
    Ok(())
}
