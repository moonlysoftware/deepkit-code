//! Deterministic text-enrichment selection; v1 §6.8 and §8.3.
use super::{Error, Kind, Target};
use crate::{
    config::Config,
    store::FileKind,
    treat::{Enrichment, Router},
};
use rusqlite::{Transaction, params};

/// A logical owner and its current parent occurrence, without outbound authority.
/// Dispatch must resolve current source, policy and prepared input independently.
#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub(crate) kind: Kind,
    pub(crate) target: Target,
    pub(crate) target_id: i64,
    pub(crate) parent_unit: i64,
    pub(crate) selection_score: f64,
}

/// Selects the configured text-job cohorts in a caller-owned coherent snapshot.
/// Treatment exclusions do not consume symbol slots; histories never get L3 jobs.
pub(crate) fn select(tx: &Transaction<'_>, config: &Config) -> Result<Vec<Candidate>, Error> {
    if !config.llm.enabled {
        return Ok(Vec::new());
    }
    let router = Router::new(&config.treatments, config.llm.enabled)
        .map_err(|_| Error::Invalid("enrichment treatment policy"))?;
    let mut selected = Vec::new();
    if config.enrich.top_symbols > 0 {
        let mut query = tx.prepare(
            "SELECT s.id,u.id,s.pagerank,f.path,f.kind FROM symbols s
            JOIN units u ON u.id=s.unit_id JOIN files f ON f.id=s.file_id
            WHERE u.file_id=f.id AND u.kind IN ('symbol','chunk') AND u.ladder_level>=1
            AND f.max_level>=3 AND f.l1_pending=0 AND f.l1_fingerprint IS NOT NULL
            AND f.l1_fingerprint=f.l1_desired AND f.l1_deferred_reason IS NULL
            ORDER BY s.pagerank DESC,s.id",
        )?;
        let mut rows = query.query([])?;
        let mut admitted = 0_u64;
        while admitted < config.enrich.top_symbols {
            let Some(row) = rows.next()? else {
                break;
            };
            let path: String = row.get(3)?;
            let kind: String = row.get(4)?;
            let kind = FileKind::from_db(&kind).ok_or(Error::Invalid("enrichment file kind"))?;
            let treatment = router.route(&path, kind);
            if treatment.never_send
                || treatment.max_level < 3
                || !treatment.enrich.contains(&Enrichment::DescribeSymbol)
            {
                continue;
            }
            selected.push(Candidate {
                kind: Kind::DescribeSymbol,
                target: Target::Symbol,
                target_id: row.get(0)?,
                parent_unit: row.get(1)?,
                selection_score: row.get(2)?,
            });
            admitted += 1;
        }
    }
    if config.commits.prs && config.enrich.prs_last_n > 0 {
        let mut query = tx.prepare(
            "SELECT p.id,u.id,p.merged_at FROM prs p JOIN units u ON u.id=p.unit_id
            WHERE u.kind='pr' AND u.file_id IS NULL AND u.ladder_level>=1
            ORDER BY p.merged_at DESC,p.id DESC LIMIT ?1",
        )?;
        let rows = query.query_map(
            [config.enrich.prs_last_n.min(i64::MAX as u64) as i64],
            |row| {
                Ok(Candidate {
                    kind: Kind::SummarizePr,
                    target: Target::Pr,
                    target_id: row.get(0)?,
                    parent_unit: row.get(1)?,
                    selection_score: row.get::<_, i64>(2)? as f64,
                })
            },
        )?;
        selected.extend(rows.collect::<Result<Vec<_>, _>>()?);
    }
    // A module's rank is the highest member file rank, as in the embedding
    // catalogue. Count only subtree members, not external graph contributors.
    let mut query = tx.prepare(
        "SELECT d.module_unit,d.file_count*coalesce(max(f.rank),0.0) AS score
        FROM dirs d JOIN units u ON u.id=d.module_unit
        LEFT JOIN files f ON d.path='' OR substr(f.path,1,length(d.path)+1)=d.path||'/'
        WHERE d.file_count>=?1 AND u.kind='module' AND u.file_id IS NULL AND u.ladder_level>=1
        GROUP BY d.path,d.module_unit,d.file_count ORDER BY score DESC,d.path",
    )?;
    let rows = query.query_map(
        params![config.enrich.modules_min_files.min(i64::MAX as u64) as i64],
        |row| {
            let parent_unit = row.get(0)?;
            Ok(Candidate {
                kind: Kind::SummarizeModule,
                target: Target::Unit,
                target_id: parent_unit,
                parent_unit,
                selection_score: row.get(1)?,
            })
        },
    )?;
    selected.extend(rows.collect::<Result<Vec<_>, _>>()?);
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_selection_keeps_logical_owners_and_applies_caps_and_treatment()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_root, store, mut config) = crate::queue::source::tests::fixture()?;
        config.llm.enabled = true;
        config.enrich.top_symbols = 2;
        config.treatments = vec![crate::config::Treatment {
            name: "source".to_owned(),
            patterns: vec!["**".to_owned()],
            max_level: 3,
            enrich: vec!["describe_symbol".to_owned()],
            ..Default::default()
        }];
        let mut writer = store.writer()?;
        let tx = writer.transaction()?;
        for (id, rank) in [(1, 0.2), (2, 0.9), (3, 0.5)] {
            tx.execute(
                "INSERT INTO symbols(id,file_id,unit_id,name,kind,line_start,line_end,pagerank)
                VALUES(?1,1,1,?2,'function',1,1,?3)",
                params![id, format!("symbol{id}"), rank],
            )?;
        }
        let selected = select(&tx, &config)?;
        assert_eq!(
            selected.iter().map(|c| c.target_id).collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert!(selected.iter().all(|c| c.parent_unit == 1
            && matches!(c.target, Target::Symbol)
            && matches!(c.kind, Kind::DescribeSymbol)));
        assert!(selected[0].selection_score > selected[1].selection_score);
        tx.execute("UPDATE files SET l1_pending=1", [])?;
        assert!(select(&tx, &config)?.is_empty());
        tx.execute("UPDATE files SET max_level=1", [])?;
        tx.execute("UPDATE files SET l1_pending=0", [])?;
        assert!(select(&tx, &config)?.is_empty());
        tx.execute("UPDATE files SET max_level=3", [])?;
        tx.execute("UPDATE files SET l1_pending=0", [])?;
        config.treatments = toml::from_str::<Config>(
            "[[treatments]]\nname='private'\nmatch=['**']\nmax_level=3\nnever_send=true\n",
        )?
        .treatments;
        assert!(select(&tx, &config)?.is_empty());
        config.treatments[0].never_send = false;
        assert!(
            select(&tx, &config)?.is_empty(),
            "an empty enrich list must refuse"
        );
        config.treatments[0].enrich = vec!["describe_symbol".to_owned()];
        assert_eq!(select(&tx, &config)?.len(), 2);
        config.llm.enabled = false;
        assert!(select(&tx, &config)?.is_empty());
        Ok(())
    }

    #[test]
    fn composite_selection_orders_pr_dates_and_module_size_times_member_rank()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_root, store, mut config) = crate::queue::source::tests::fixture()?;
        config.llm.enabled = true;
        config.enrich.top_symbols = 0;
        config.enrich.prs_last_n = 2;
        config.enrich.modules_min_files = 5;
        let mut writer = store.writer()?;
        let tx = writer.transaction()?;
        for (id, kind) in [
            (11, "pr"),
            (12, "pr"),
            (13, "pr"),
            (21, "module"),
            (22, "module"),
            (23, "module"),
            (30, "history"),
        ] {
            let hash = blake3::hash(format!("{kind}:{id}").as_bytes());
            tx.execute("INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at)
                SELECT ?1,?2,content_hash,?3,?4,'fixture',0,0 FROM units WHERE id=1",params![id,hash.as_bytes().as_slice(),format!("{kind}:{id}"),kind])?;
        }
        for (id, merged_at) in [(11, 100), (12, 300), (13, 200)] {
            tx.execute(
                "INSERT INTO prs(id,title,merge_sha,strategy,commit_count,merged_at,unit_id)
                VALUES(?1,'fixture',?2,'merge',1,?3,?1)",
                params![id, vec![id as u8; 20], merged_at],
            )?;
        }
        for (id, path, count, rank) in [(21, "a", 5, 0.9), (22, "b", 10, 0.2), (23, "c", 4, 1.0)] {
            tx.execute("INSERT INTO dirs(path,merkle,file_count,module_unit) VALUES(?1,zeroblob(32),?2,?3)",params![path,count,id])?;
            tx.execute(
                "INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment,rank)
                VALUES(?1,?2,zeroblob(32),0,0,'code',3,'source',?3)",
                params![id, format!("{path}/file.rs"), rank],
            )?;
        }
        let rows = select(&tx, &config)?;
        assert_eq!(
            rows.iter().map(|c| c.target_id).collect::<Vec<_>>(),
            vec![12, 13, 21, 22]
        );
        assert!(
            rows[..2]
                .iter()
                .all(|c| matches!(c.kind, Kind::SummarizePr) && matches!(c.target, Target::Pr))
        );
        assert!(
            rows[2..].iter().all(
                |c| matches!(c.kind, Kind::SummarizeModule) && matches!(c.target, Target::Unit)
            )
        );
        assert_eq!(rows[2].selection_score, 4.5);
        assert_eq!(rows[3].selection_score, 2.0);
        config.commits.prs = false;
        assert_eq!(
            select(&tx, &config)?
                .iter()
                .map(|c| c.target_id)
                .collect::<Vec<_>>(),
            vec![21, 22]
        );
        Ok(())
    }
}
