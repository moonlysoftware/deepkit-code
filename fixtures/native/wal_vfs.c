/* Native WAL VFS memory/UB and bypass checks; v1 §3.5. Generated DB in /tmp. */
#include <sqlite3.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
int dkc_wal_vfs_register(void);
static void sql(sqlite3 *db, const char *s) {
    int rc = sqlite3_exec(db, s, 0, 0, 0);
    if (rc != SQLITE_OK) fprintf(stderr, "fixture SQL failed (%d): %s\n", rc, sqlite3_errmsg(db));
    assert(rc == SQLITE_OK);
}
int main(int argc, char **argv) {
    sqlite3 *db = 0;
    sqlite3_file *wal = 0;
    sqlite3_int64 size = -1, hint = 1048576;
    int chunk = 1048576;
    char bytes[32] = {0};
    assert(argc == 2);
    assert(dkc_wal_vfs_register() == SQLITE_OK);
    assert(sqlite3_open_v2(argv[1], &db, SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE, "dkc-wal") == SQLITE_OK);
    sql(db, "PRAGMA journal_mode=WAL; CREATE TABLE t(x); INSERT INTO t VALUES(1); PRAGMA wal_checkpoint(TRUNCATE)");
    assert(sqlite3_file_control(db, "main", SQLITE_FCNTL_JOURNAL_POINTER, &wal) == SQLITE_OK);
    assert(wal && wal->pMethods);
    sql(db, "PRAGMA dkc_wal_limit=8192");
    /* Neither preallocation nor chunk rounding may bypass the WAL ceiling. */
    assert(wal->pMethods->xFileControl(wal, SQLITE_FCNTL_SIZE_HINT, &hint) == SQLITE_OK);
    assert(wal->pMethods->xFileSize(wal, &size) == SQLITE_OK && size == 0);
    assert(wal->pMethods->xFileControl(wal, SQLITE_FCNTL_CHUNK_SIZE, &chunk) == SQLITE_OK);
    assert(wal->pMethods->xTruncate(wal, 4096) == SQLITE_OK);
    assert(wal->pMethods->xFileSize(wal, &size) == SQLITE_OK && size == 4096);
    assert(wal->pMethods->xTruncate(wal, 8193) == SQLITE_FULL);
    assert(wal->pMethods->xWrite(wal, bytes, 32, 8192) == SQLITE_FULL);
    assert(wal->pMethods->xWrite(wal, bytes, 32, INT64_MAX) == SQLITE_FULL);
    assert(wal->pMethods->xWrite(wal, bytes, 32, -1) == SQLITE_FULL);
    assert(wal->pMethods->xFileSize(wal, &size) == SQLITE_OK && size == 4096);
    /* Clean up this direct-method fixture before returning to ordinary SQLite. */
    assert(wal->pMethods->xTruncate(wal, 0) == SQLITE_OK);
    sql(db, "PRAGMA dkc_wal_limit=9223372036854775807; UPDATE t SET x=2; PRAGMA wal_checkpoint(TRUNCATE)");
    assert(sqlite3_close(db) == SQLITE_OK);
    db = 0;
    /* Opening an absent parent exercises allocation/close cleanup on xOpen failure. */
    assert(sqlite3_open_v2("/nonexistent-dkc-wal-fixture/db", &db, SQLITE_OPEN_READWRITE, "dkc-wal") == SQLITE_CANTOPEN);
    assert(sqlite3_close(db) == SQLITE_OK);
    assert(sqlite3_shutdown() == SQLITE_OK);
    puts("native WAL VFS: size hints, chunking, truncation, overflow, failure cleanup passed");
    return 0;
}
