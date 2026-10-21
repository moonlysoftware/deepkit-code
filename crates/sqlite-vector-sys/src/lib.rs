//! Native sqlite-vector and bounded WAL VFS bindings; see `v1.md` §3.1/§3.5.

use std::ffi::{c_char, c_int};
use std::{error::Error, fmt, sync::OnceLock};

// SAFETY: These declarations match the SQLite extension ABI and our no-argument
// VFS registration entry point; both static libraries are built by this crate.
unsafe extern "C" {
    fn dkc_wal_vfs_register() -> c_int;
    fn sqlite3_dkc_vector_init(
        db: *mut libsqlite3_sys::sqlite3,
        error_message: *mut *mut c_char,
        api: *const libsqlite3_sys::sqlite3_api_routines,
    ) -> c_int;
    /// Registers sqlite-vector functions and virtual tables on one connection.
    pub fn sqlite3_vector_init(
        db: *mut libsqlite3_sys::sqlite3,
        error_message: *mut *mut c_char,
        api: *const libsqlite3_sys::sqlite3_api_routines,
    ) -> c_int;
}

/// An error returned when SQLite rejects extension or VFS registration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistrationError {
    code: c_int,
}

impl fmt::Display for RegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "SQLite rejected native extension/VFS registration (code {})",
            self.code
        )
    }
}

/// Registers the named WAL-ceiling VFS without replacing SQLite's default VFS.
/// Call before opening a connection with the returned name. The ceiling remains
/// disabled until that connection sets the internal `dkc_wal_limit` pragma.
pub fn register_wal_vfs() -> Result<&'static str, RegistrationError> {
    static REGISTRATION: OnceLock<Result<(), RegistrationError>> = OnceLock::new();
    (*REGISTRATION.get_or_init(|| {
        // SAFETY: This statically linked C function takes no pointers. OnceLock
        // serializes registration; the VFS and delegated platform VFS remain live.
        let code = unsafe { dkc_wal_vfs_register() };
        if code == 0 {
            Ok(())
        } else {
            Err(RegistrationError { code })
        }
    }))?;
    Ok("dkc-wal")
}

impl Error for RegistrationError {}

/// Registers sqlite-vector functions without writing tables during connection open.
/// Call `SELECT dkc_vector_metadata_init()` in an admitted transaction before
/// registering vector tables; the native metadata schema remains upstream-owned.
///
/// The FFI call stays in this `-sys` crate so the store layer has a safe API.
pub fn register_auto_extension() -> Result<(), RegistrationError> {
    static REGISTRATION: OnceLock<Result<(), RegistrationError>> = OnceLock::new();

    *REGISTRATION.get_or_init(|| {
        // SAFETY: `sqlite3_dkc_vector_init` has SQLite's documented extension ABI,
        // remains linked for the process lifetime, and SQLite de-duplicates it.
        let code = unsafe { libsqlite3_sys::sqlite3_auto_extension(Some(sqlite3_dkc_vector_init)) };
        if code == 0 {
            Ok(())
        } else {
            Err(RegistrationError { code })
        }
    })
}
