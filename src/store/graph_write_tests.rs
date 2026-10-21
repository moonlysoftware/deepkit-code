//! Checked graph publication refuses cancelled writer waiters without clearing pending work.
use super::{Error, GraphFileRank, GraphSymbolRank, Store};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

fn ranks(store: &Store) -> (f64, f64, f64, String) {
    store
        .with_reader(|reader| {
            reader
                .query_row(
                    "SELECT (SELECT pagerank FROM symbols WHERE id=1),
                    (SELECT rank FROM files WHERE id=1),
                    (SELECT rank FROM files WHERE id=2),
                    (SELECT value FROM meta WHERE key='graph_pending')",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .map_err(Error::from)
        })
        .unwrap()
}

#[test]
fn cancelled_graph_writer_preserves_ranks_pending_and_retry() {
    for file_backed in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let path = if file_backed {
            root.path().join("code.db")
        } else {
            ":memory:".into()
        };
        let store = super::open(path, Default::default()).unwrap();
        store
            .writer()
            .unwrap()
            .execute_batch(
                "INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment,rank)
             VALUES(1,'source.rs',zeroblob(32),1,0,'code',1,'source',0.25),
                   (2,'empty.rs',zeroblob(32),0,0,'code',1,'source',0.75);
             INSERT INTO symbols(id,file_id,name,kind,line_start,line_end,pagerank)
             VALUES(1,1,'source','function',1,1,0.25);
             INSERT INTO meta(key,value) VALUES('graph_pending','1')
             ON CONFLICT(key) DO UPDATE SET value='1';",
            )
            .unwrap();
        let generation = store.l1_generation().unwrap();
        let before = ranks(&store);
        let symbols = [GraphSymbolRank {
            symbol_id: 1,
            rank: 0.9,
        }];
        let files = [GraphFileRank {
            file_id: 1,
            rank: 0.9,
        }];
        let cancelled = AtomicBool::new(false);
        let observed = AtomicBool::new(false);
        let held = store.writer().unwrap();
        std::thread::scope(|scope| {
            let (entered, started) = mpsc::channel();
            let (finished, done) = mpsc::channel();
            let store = &store;
            let symbols = &symbols;
            let files = &files;
            let cancelled = &cancelled;
            let observed = &observed;
            scope.spawn(move || {
                let check = || {
                    if !observed.swap(true, Ordering::AcqRel) {
                        entered.send(()).unwrap();
                    }
                    if cancelled.load(Ordering::Acquire) {
                        Err(Error::L1InputsChanged)
                    } else {
                        Ok(())
                    }
                };
                finished
                    .send(store.update_graph_ranks_checked(
                        generation,
                        symbols,
                        files,
                        Some(&check),
                    ))
                    .unwrap();
            });
            let admitted = started.recv_timeout(Duration::from_secs(2));
            cancelled.store(true, Ordering::Release);
            let result = done.recv_timeout(Duration::from_secs(2));
            // The writer is held until the waiter returns; cancellation cannot
            // depend on the next successful lock acquisition.
            drop(held);
            admitted.unwrap();
            assert!(matches!(result.unwrap(), Err(Error::L1InputsChanged)));
        });
        assert_eq!(ranks(&store), before);
        assert_eq!(store.l1_generation().unwrap(), generation);
        store
            .update_graph_ranks_checked(generation, &symbols, &files, Some(&|| Ok(())))
            .unwrap();
        assert_eq!(ranks(&store), (0.9, 0.9, 0.0, "0".into()));
        // Existing unbounded wrapper keeps the same mutation semantics.
        store
            .update_graph_ranks(
                generation,
                &[GraphSymbolRank {
                    symbol_id: 1,
                    rank: 0.4,
                }],
                &[GraphFileRank {
                    file_id: 1,
                    rank: 0.4,
                }],
            )
            .unwrap();
        assert_eq!(ranks(&store), (0.4, 0.4, 0.0, "0".into()));
    }
}
