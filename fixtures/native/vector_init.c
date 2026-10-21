/* Deferred/eager extension startup and metadata lifecycle; v1 §3.5. */
#include <assert.h>
#include <stdio.h>
#include <sqlite3.h>

extern int sqlite3_vector_init(sqlite3 *, char **, const sqlite3_api_routines *);
extern int sqlite3_dkc_vector_init(sqlite3 *, char **, const sqlite3_api_routines *);

static void sql(sqlite3 *db, const char *text) {
    int rc = sqlite3_exec(db, text, NULL, NULL, NULL);
    if (rc != SQLITE_OK) fprintf(stderr, "SQLite: %s\n", sqlite3_errmsg(db));
    assert(rc == SQLITE_OK);
}

static int scalar(sqlite3 *db, const char *text) {
    sqlite3_stmt *stmt = NULL;
    assert(sqlite3_prepare_v2(db, text, -1, &stmt, NULL) == SQLITE_OK);
    assert(sqlite3_step(stmt) == SQLITE_ROW);
    int value = sqlite3_column_int(stmt, 0);
    assert(sqlite3_finalize(stmt) == SQLITE_OK);
    return value;
}

/* Keep a live cursor while nested reclamation is attempted. Its table context
 * must survive both the rejected call and subsequent cursor advancement. */
static void lifecycle(sqlite3 *db) {
    sqlite3_stmt *scan = NULL;
    assert(sqlite3_prepare_v2(db, "SELECT rowid FROM vector_full_scan('embeddings','v',vector_as_f16('[1,0,0,0]'))", -1, &scan, NULL) == SQLITE_OK);
    assert(sqlite3_step(scan) == SQLITE_ROW);
    assert(sqlite3_exec(db, "SELECT dkc_vector_prune()", NULL, NULL, NULL) == SQLITE_BUSY);
    assert(sqlite3_step(scan) == SQLITE_ROW);
    assert(sqlite3_finalize(scan) == SQLITE_OK);
    sql(db, "BEGIN; DROP TABLE embeddings;");
    assert(sqlite3_exec(db, "SELECT dkc_vector_prune()", NULL, NULL, NULL) == SQLITE_BUSY);
    sql(db, "ROLLBACK;");
    assert(scalar(db, "SELECT dkc_vector_prune()") == 0);
    assert(scalar(db, "SELECT rowid FROM vector_full_scan('embeddings','v',vector_as_f16('[1,0,0,0]'),1)") == 1);
    for (int i = 0; i < 256; i++) {
        char stmt[512];
        snprintf(stmt, sizeof(stmt), "CREATE TABLE obsolete_%d(v BLOB); SELECT vector_init('obsolete_%d','v','dimension=2,type=FLOAT16,distance=COSINE'); DROP TABLE obsolete_%d;", i, i, i);
        sql(db, stmt);
        assert(scalar(db, "SELECT dkc_vector_prune()") == 1);
        assert(scalar(db, "SELECT dkc_vector_prune()") == 0);
    }
    assert(scalar(db, "SELECT rowid FROM vector_full_scan('embeddings','v',vector_as_f16('[1,0,0,0]'),1)") == 1);
}

int main(void) {
    for (int deferred = 0; deferred <= 1; deferred++) {
        sqlite3_reset_auto_extension();
        assert(sqlite3_auto_extension((void (*)(void))(deferred ? sqlite3_dkc_vector_init : sqlite3_vector_init)) == SQLITE_OK);
        for (int repeat = 0; repeat < 64; repeat++) {
            sqlite3 *db = NULL;
            assert(sqlite3_open(":memory:", &db) == SQLITE_OK);
            sql(db, "SELECT vector_version()");
            const char *exists = "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE name='_sqliteai_vector')";
            assert(scalar(db, exists) == !deferred);
            if (deferred) {
                sql(db, "BEGIN; SELECT dkc_vector_metadata_init(); ROLLBACK;");
                assert(scalar(db, exists) == 0);
                sql(db, "BEGIN; SELECT dkc_vector_metadata_init(); COMMIT;");
                assert(scalar(db, exists) == 1);
            }
            sql(db, "CREATE TABLE embeddings(v BLOB);"
                "SELECT vector_init('embeddings','v','dimension=4,type=FLOAT16,distance=COSINE');"
                "INSERT INTO embeddings VALUES(vector_as_f16('[1,0,0,0]'));"
                "INSERT INTO embeddings VALUES(vector_as_f16('[0,1,0,0]'));");
            assert(scalar(db, "SELECT rowid FROM embeddings WHERE rowid=1") == 1);
            assert(scalar(db, "SELECT rowid FROM vector_full_scan('embeddings','v',vector_as_f16('[1,0,0,0]'),1)") == 1);
            if (repeat == 0) lifecycle(db);
            assert(sqlite3_close(db) == SQLITE_OK);
        }
    }
    sqlite3_reset_auto_extension();
    assert(sqlite3_shutdown() == SQLITE_OK);
    puts("eager/deferred initialization, metadata rollback and vector lifetime: 128 opens pass");
    return 0;
}
