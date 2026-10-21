use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=src/wal_vfs.c");
    let sqlite_include =
        PathBuf::from(env::var("DEP_SQLITE3_INCLUDE").expect("bundled SQLite headers"));
    cc::Build::new()
        .include(&sqlite_include)
        .file("src/wal_vfs.c")
        .warnings(true)
        .extra_warnings(true)
        .compile("dkc_wal_vfs");
    let vendor = PathBuf::from("vendor/sqlite-vector");
    let src = vendor.join("src");
    let libs = vendor.join("libs");

    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed={}", libs.display());

    let mut baseline = build(&src, &libs, &sqlite_include);
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        // MSVC's CRT exposes these POSIX comparisons under Windows names.
        baseline
            .define("strcasecmp", Some("_stricmp"))
            .define("strncasecmp", Some("_strnicmp"));
    }
    baseline
        .file(src.join("sqlite-vector.c"))
        .file(src.join("distance-cpu.c"))
        .file(src.join("distance-sse2.c"))
        .file(src.join("distance-neon.c"))
        .file(src.join("distance-rvv.c"))
        .compile("sqlite_vector_baseline");

    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64") {
        let mut avx2 = build(&src, &libs, &sqlite_include);
        avx2.flag_if_supported("-mavx2")
            .flag_if_supported("-mfma")
            .flag_if_supported("-mf16c")
            .file(src.join("distance-avx2.c"))
            .compile("sqlite_vector_avx2");

        let mut avx512 = build(&src, &libs, &sqlite_include);
        avx512
            .flag_if_supported("-mavx512f")
            .flag_if_supported("-mavx512bw")
            .flag_if_supported("-mavx512vl")
            .flag_if_supported("-mavx512dq")
            .file(src.join("distance-avx512.c"))
            .compile("sqlite_vector_avx512");
    }
}

fn build(
    src: &std::path::Path,
    libs: &std::path::Path,
    sqlite_include: &std::path::Path,
) -> cc::Build {
    let mut build = cc::Build::new();
    build
        .include(src)
        .include(libs)
        .include(sqlite_include)
        .warnings(true)
        .extra_warnings(true)
        .flag_if_supported("-Wno-unused-parameter")
        .opt_level(2);
    build
}
