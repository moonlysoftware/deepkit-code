/* WAL write-boundary ceilings; v1 §3.5. MIT OR Apache-2.0.
 * This named, non-default VFS delegates locking, SHM, reads and durability to
 * SQLite's platform VFS. No files or locks of its own. Register once before use.
 * SQLite serializes each connection; its main file outlives its WAL handle.
 */
#include <sqlite3.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

typedef struct DkcFile {
    sqlite3_file base;
    sqlite3_file *real;
    struct DkcFile *main;
    sqlite3_int64 limit;
    int hit;
    int wal;
} DkcFile;
static sqlite3_vfs vfs;
static const sqlite3_io_methods methods;
static sqlite3_vfs *parent(sqlite3_vfs *v) { return (sqlite3_vfs *)v->pAppData; }
static DkcFile *file(sqlite3_file *f) { return (DkcFile *)f; }
static sqlite3_file *real(sqlite3_file *f) { return file(f)->real; }

static int close_file(sqlite3_file *f) {
    int rc = real(f)->pMethods->xClose(real(f));
    sqlite3_free(real(f));
    f->pMethods = 0;
    return rc;
}
static int read_file(sqlite3_file *f, void *b, int n, sqlite3_int64 off) {
    return real(f)->pMethods->xRead(real(f), b, n, off);
}
static int permits(sqlite3_file *f, sqlite3_int64 off, sqlite3_int64 n) {
    DkcFile *d = file(f);
    if (!d->wal) return SQLITE_OK;
    if (off < 0 || n < 0 || off > d->main->limit || n > d->main->limit - off) {
        d->main->hit = 1;
        return SQLITE_FULL;
    }
    return SQLITE_OK;
}
static int write_file(sqlite3_file *f, const void *b, int n, sqlite3_int64 off) {
    int rc = permits(f, off, n);
    return rc == SQLITE_OK ? real(f)->pMethods->xWrite(real(f), b, n, off) : rc;
}
static int truncate_file(sqlite3_file *f, sqlite3_int64 n) {
    int rc = permits(f, 0, n);
    return rc == SQLITE_OK ? real(f)->pMethods->xTruncate(real(f), n) : rc;
}
static int sync_file(sqlite3_file *f, int flags) { return real(f)->pMethods->xSync(real(f), flags); }
static int size_file(sqlite3_file *f, sqlite3_int64 *n) { return real(f)->pMethods->xFileSize(real(f), n); }
static int lock_file(sqlite3_file *f, int n) { return real(f)->pMethods->xLock(real(f), n); }
static int unlock_file(sqlite3_file *f, int n) { return real(f)->pMethods->xUnlock(real(f), n); }
static int reserved(sqlite3_file *f, int *n) { return real(f)->pMethods->xCheckReservedLock(real(f), n); }
static int sector(sqlite3_file *f) { return real(f)->pMethods->xSectorSize(real(f)); }
static int characteristics(sqlite3_file *f) { return real(f)->pMethods->xDeviceCharacteristics(real(f)); }

/* Parse the whole nonnegative decimal string, rejecting signs and overflow. */
static int decimal(const char *s, sqlite3_int64 *result) {
    sqlite3_int64 n = 0;
    if (!s || !*s) return SQLITE_ERROR;
    for (; *s; ++s) {
        if (*s < '0' || *s > '9' || n > (INT64_MAX - (*s - '0')) / 10) return SQLITE_ERROR;
        n = n * 10 + (*s - '0');
    }
    *result = n;
    return SQLITE_OK;
}
static int control(sqlite3_file *f, int op, void *arg) {
    DkcFile *d = file(f);
    /* Hints may preallocate without xWrite, and chunking can round xTruncate
     * above its requested size. Suppress both for WAL even before a cap is set. */
    if (d->wal && (op == SQLITE_FCNTL_SIZE_HINT || op == SQLITE_FCNTL_CHUNK_SIZE)) return SQLITE_OK;
    if (op == SQLITE_FCNTL_PRAGMA && d->main == d) {
        char **a = (char **)arg;
        if (sqlite3_stricmp(a[1], "dkc_wal_limit") == 0) {
            if (a[2]) {
                sqlite3_int64 n;
                if (decimal(a[2], &n) != SQLITE_OK) {
                    a[0] = sqlite3_mprintf("invalid WAL limit");
                    return a[0] ? SQLITE_ERROR : SQLITE_NOMEM;
                }
                d->limit = n;
                d->hit = 0;
            }
            a[0] = sqlite3_mprintf("%lld", (long long)d->limit);
            return a[0] ? SQLITE_OK : SQLITE_NOMEM;
        }
        if (sqlite3_stricmp(a[1], "dkc_wal_limit_hit") == 0 && !a[2]) {
            a[0] = sqlite3_mprintf("%d", d->hit);
            return a[0] ? SQLITE_OK : SQLITE_NOMEM;
        }
    }
    return real(f)->pMethods->xFileControl(real(f), op, arg);
}
static int shm_map(sqlite3_file *f, int pg, int n, int extend, void volatile **out) {
    if (real(f)->pMethods->iVersion < 2 || !real(f)->pMethods->xShmMap) return SQLITE_IOERR_SHMMAP;
    return real(f)->pMethods->xShmMap(real(f), pg, n, extend, out);
}
static int shm_lock(sqlite3_file *f, int off, int n, int flags) {
    if (real(f)->pMethods->iVersion < 2 || !real(f)->pMethods->xShmLock) return SQLITE_IOERR_SHMLOCK;
    return real(f)->pMethods->xShmLock(real(f), off, n, flags);
}
static void shm_barrier(sqlite3_file *f) {
    if (real(f)->pMethods->iVersion >= 2 && real(f)->pMethods->xShmBarrier) real(f)->pMethods->xShmBarrier(real(f));
}
static int shm_unmap(sqlite3_file *f, int del) {
    if (real(f)->pMethods->iVersion < 2 || !real(f)->pMethods->xShmUnmap) return SQLITE_OK;
    return real(f)->pMethods->xShmUnmap(real(f), del);
}
static int fetch(sqlite3_file *f, sqlite3_int64 off, int n, void **out) {
    *out = 0;
    if (real(f)->pMethods->iVersion < 3 || !real(f)->pMethods->xFetch) return SQLITE_OK;
    return real(f)->pMethods->xFetch(real(f), off, n, out);
}
static int unfetch(sqlite3_file *f, sqlite3_int64 off, void *p) {
    if (real(f)->pMethods->iVersion < 3 || !real(f)->pMethods->xUnfetch) return SQLITE_OK;
    return real(f)->pMethods->xUnfetch(real(f), off, p);
}
static const sqlite3_io_methods methods = {
    3, close_file, read_file, write_file, truncate_file, sync_file, size_file,
    lock_file, unlock_file, reserved, control, sector, characteristics,
    shm_map, shm_lock, shm_barrier, shm_unmap, fetch, unfetch
};
static int open_file(sqlite3_vfs *v, const char *name, sqlite3_file *f, int flags, int *out) {
    DkcFile *d = file(f);
    int rc;
    memset(d, 0, sizeof(*d));
    d->limit = INT64_MAX;
    d->wal = (flags & SQLITE_OPEN_WAL) != 0;
    if (flags & SQLITE_OPEN_MAIN_DB) d->main = d;
    if (d->wal) {
        /* name is exactly SQLite's WAL xOpen filename; never a copied string. */
        sqlite3_file *db = sqlite3_database_file_object(name);
        if (!db || db->pMethods != &methods) return SQLITE_CANTOPEN;
        d->main = file(db);
    }
    d->real = sqlite3_malloc64((sqlite3_uint64)parent(v)->szOsFile);
    if (!d->real) return SQLITE_NOMEM;
    memset(d->real, 0, (size_t)parent(v)->szOsFile);
    rc = parent(v)->xOpen(parent(v), name, d->real, flags, out);
    if (rc != SQLITE_OK) {
        if (d->real->pMethods) d->real->pMethods->xClose(d->real);
        sqlite3_free(d->real);
        d->real = 0;
        return rc;
    }
    d->base.pMethods = &methods;
    return SQLITE_OK;
}
static int delete_file(sqlite3_vfs *v, const char *n, int sync) { return parent(v)->xDelete(parent(v), n, sync); }
static int access_file(sqlite3_vfs *v, const char *n, int flags, int *out) { return parent(v)->xAccess(parent(v), n, flags, out); }
static int full_path(sqlite3_vfs *v, const char *n, int len, char *out) { return parent(v)->xFullPathname(parent(v), n, len, out); }
static void *dl_open(sqlite3_vfs *v, const char *n) { return parent(v)->xDlOpen ? parent(v)->xDlOpen(parent(v), n) : 0; }
static void dl_error(sqlite3_vfs *v, int n, char *out) { if (parent(v)->xDlError) parent(v)->xDlError(parent(v), n, out); else if (n > 0) out[0] = 0; }
static void (*dl_sym(sqlite3_vfs *v, void *h, const char *n))(void) { return parent(v)->xDlSym ? parent(v)->xDlSym(parent(v), h, n) : 0; }
static void dl_close(sqlite3_vfs *v, void *h) { if (parent(v)->xDlClose) parent(v)->xDlClose(parent(v), h); }
static int randomness(sqlite3_vfs *v, int n, char *out) { return parent(v)->xRandomness(parent(v), n, out); }
static int sleep_vfs(sqlite3_vfs *v, int n) { return parent(v)->xSleep(parent(v), n); }
static int time_vfs(sqlite3_vfs *v, double *out) { return parent(v)->xCurrentTime(parent(v), out); }
static int last_error(sqlite3_vfs *v, int n, char *out) { return parent(v)->xGetLastError ? parent(v)->xGetLastError(parent(v), n, out) : 0; }
static int time64(sqlite3_vfs *v, sqlite3_int64 *out) { return parent(v)->xCurrentTimeInt64(parent(v), out); }

/* Called through Rust OnceLock; this static VFS and the platform VFS live for
 * the process lifetime. We intentionally do not replace SQLite's default VFS. */
int dkc_wal_vfs_register(void) {
    sqlite3_vfs *p = sqlite3_vfs_find(0);
    if (!p || p->iVersion < 2 || !p->xCurrentTimeInt64) return SQLITE_ERROR;
    memset(&vfs, 0, sizeof(vfs));
    vfs.iVersion = 2;
    vfs.szOsFile = sizeof(DkcFile);
    vfs.mxPathname = p->mxPathname;
    vfs.zName = "dkc-wal";
    vfs.pAppData = p;
    vfs.xOpen = open_file; vfs.xDelete = delete_file; vfs.xAccess = access_file;
    vfs.xFullPathname = full_path; vfs.xDlOpen = dl_open; vfs.xDlError = dl_error;
    vfs.xDlSym = dl_sym; vfs.xDlClose = dl_close; vfs.xRandomness = randomness;
    vfs.xSleep = sleep_vfs; vfs.xCurrentTime = time_vfs; vfs.xGetLastError = last_error;
    vfs.xCurrentTimeInt64 = time64;
    return sqlite3_vfs_register(&vfs, 0);
}
