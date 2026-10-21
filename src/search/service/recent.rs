//! Shared bounded recent history query and immutable-source fences; v1.md §10.1.
use super::*;
use rusqlite::{OptionalExtension, params};
use std::collections::BTreeSet;

/// Identical filters for CLI and MCP; an absent granularity uses repository settings.
#[derive(Clone, Default)]
pub(crate) struct Request {
    pub(crate) path: Option<String>,
    pub(crate) symbol: Option<String>,
    pub(crate) since: Option<String>,
    pub(crate) granularity: Option<String>,
    pub(crate) k: Option<usize>,
}
fn invalid() -> Error {
    super::super::Error::InvalidQuery {
        reason: "invalid recent filter",
    }
    .into()
}
fn since(value: Option<&str>, now: i64) -> Result<i64> {
    let Some(value) = value else {
        return Ok(i64::MIN);
    };
    if let Some(days) = value.strip_suffix('d') {
        let days = days
            .parse::<i64>()
            .ok()
            .filter(|d| *d >= 0)
            .and_then(|d| d.checked_mul(86400))
            .ok_or_else(invalid)?;
        return now.checked_sub(days).ok_or_else(invalid);
    }
    if value.len() < 10
        || value.as_bytes().get(4) != Some(&b'-')
        || value.as_bytes().get(7) != Some(&b'-')
    {
        return Err(invalid());
    }
    let date = if value.len() == 10 {
        format!("{value}T00:00:00Z")
    } else {
        value.to_owned()
    };
    gix::date::parse(&date, None)
        .map(|v| v.seconds)
        .map_err(|_| invalid())
}

impl Context {
    /// Returns verified PRs by default, with uncovered commits as fallback. Reads
    /// HEAD without loading Git's index; changed/unindexed history refuses the array.
    pub(crate) async fn recent(&self, request: Request) -> Result<Value> {
        let owner = QueryWork::new(Instant::now(), self.config.limits.query_timeout_ms)?;
        let context = self.clone();
        let operation=owner.blocking(move |work| {
            let granularity=request.granularity.as_deref().unwrap_or(&context.config.search.recent_granularity);
            if !matches!(granularity,"pr"|"commit") || request.k==Some(0) {return Err(invalid());}
            if request.path.as_deref().is_some_and(|p|p.is_empty()||p.starts_with('/')||p.split('/').any(|s|s=="..")) || request.symbol.as_deref()==Some("") {return Err(invalid());}
            let since=since(request.since.as_deref(),unix_seconds()?)?;
            let k=request.k.unwrap_or(context.config.search.default_k as usize).min(context.config.search.max_k as usize).min(50);
            if k==0 {return Err(invalid());}
            let _session=context.query_session_checked(work)?;
            let check=||work.check().map_err(|_|store::Error::L1InputsChanged);
            let head=sync::head::read(&context.root,work)?.map(|h|h.id);
            let (indexed,has_history)=context.store.checked_read_snapshot(&check,|tx| {
                let indexed=tx.query_row("SELECT value FROM meta WHERE key='indexed_head'",[],|r|r.get::<_,String>(0)).optional()?;
                let has_history=tx.query_row("SELECT EXISTS(SELECT 1 FROM meta WHERE key='history_status')",[],|r|r.get::<_,bool>(0))?;
                Ok::<_,store::Error>((indexed,has_history))
            })??;
            if !context.config.commits.enabled || !has_history {return Err(Error::HistoryNotIndexed);}
            if head!=indexed {return Err(super::super::Error::OpenSourceUnavailable.into());}
            super::super::repair::validate_configuration(&context,work)?;
            let generation=query_generation(&context.store,work)?;
            let collect=|| -> Result<(Value,bool,Vec<Option<super::super::summaries::Observation>>)> {
                context.store.checked_read_snapshot(&check,|tx| {
                    let prs=granularity=="pr"&&context.config.commits.prs;
                    let mut q=tx.prepare("WITH eligible AS (
                      SELECT c.id FROM commits c WHERE
                      (?1 IS NULL OR EXISTS(SELECT 1 FROM json_each(c.files_touched) j WHERE j.value=?1 OR substr(j.value,1,length(?1)+1)=?1||'/'))
                      AND (?2 IS NULL OR EXISTS(SELECT 1 FROM json_each(c.symbols_touched) j WHERE j.value=?2 OR substr(j.value,-length(?2)-2)='::'||?2 OR substr(j.value,-length(?2)-1)='.'||?2))
                    ), candidates AS (
                      SELECT p.unit_id AS unit,'pr' AS kind,p.title,p.merged_at AS time,NULL AS author,p.commit_count AS commits,p.id AS history_id
                      FROM prs p WHERE ?3 AND p.unit_id IS NOT NULL AND p.merged_at>=?4 AND EXISTS(SELECT 1 FROM commits c JOIN eligible e ON e.id=c.id WHERE c.pr_id=p.id)
                      UNION ALL SELECT c.unit_id,'commit',c.subject,c.author_time,c.author,NULL,c.id FROM commits c JOIN eligible e ON e.id=c.id
                      WHERE c.unit_id IS NOT NULL AND c.author_time>=?4 AND (NOT ?3 OR (c.pr_id IS NULL AND NOT EXISTS(SELECT 1 FROM prs p WHERE p.merge_sha=c.sha)))
                    ) SELECT unit,kind,title,time,author,commits,history_id FROM candidates ORDER BY time DESC,kind,unit DESC LIMIT ?5")?;
                    let rows=q.query_map(params![request.path,request.symbol,prs,since,k as i64],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,i64>(3)?,r.get::<_,Option<String>>(4)?,r.get::<_,Option<i64>>(5)?,r.get::<_,i64>(6)?)))?.collect::<std::result::Result<Vec<_>,_>>()?;
                    let mut result=Vec::new();let mut truncated=false;let mut summaries=Vec::new();
                    for (unit,kind,title,time,author,count,id) in rows {
                        check()?;
                        if queue::source::resolve_freshness(tx,&context.root,&context.config_root,&context.config,unit,work)?.is_none() {return Err(Error::Search(super::super::Error::OpenSourceUnavailable));}
                        let stored=store::unit_by_id_in(tx,unit)?.ok_or(super::super::Error::OpenSourceUnavailable)?;
                        let raw=if kind=="pr" {
                            tx.prepare("SELECT files_touched,coalesce(symbols_touched,'[]') FROM commits WHERE pr_id=?1 ORDER BY author_time,sha LIMIT 501")?.query_map([id],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?)))?.collect::<std::result::Result<Vec<_>,_>>()?
                        }else {vec![tx.query_row("SELECT files_touched,coalesce(symbols_touched,'[]') FROM commits WHERE id=?1",[id],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?)))?]};
                        let mut files=BTreeSet::new();let mut symbols=BTreeSet::new();
                        for (f,s) in raw {files.extend(serde_json::from_str::<Vec<String>>(&f)?);symbols.extend(serde_json::from_str::<Vec<String>>(&s)?);}
                        let date=gix::date::Time::new(time,0).format(gix::date::time::format::SHORT).map_err(|_|invalid())?;
                        let mut item=json!({"id":navigation::public_id(&stored),"kind":kind,"title":title,"date":date,"files":files.len(),"symbols":symbols.into_iter().take(100).collect::<Vec<_>>()});
                        if let Some(author)=author {item["author"]=json!(author);}
                        if let Some(count)=count {item["commits"]=json!(count);}
                        let summary=if kind=="pr" {super::super::summaries::observe_in(tx,&context.root,&context.config_root,&context.config,queue::summary_source::Attachment::Parent(unit),work)?} else {None};
                        if let Some(summary)=&summary {item["summary"]=json!(summary.text);}
                        if !response_limits::push(&mut result,item,context.config.limits.max_result_tokens,work)? {truncated=true;break;}
                        summaries.push(summary);
                    }
                    Ok::<_,Error>((Value::Array(result),truncated,summaries))
                })?
            };
            let (result,truncated,summaries)=collect()?;
            let (verified,verified_truncated,verified_summaries)=collect()?;
            super::super::repair::validate_configuration(&context,work)?;
            if verified!=result || truncated!=verified_truncated || summaries!=verified_summaries || query_generation(&context.store,work)?!=generation || sync::head::read(&context.root,work)?.map(|h|h.id)!=head {return Err(super::super::Error::OpenSourceUnavailable.into());}
            check_query_work(work)?;
            if truncated {context.query_latencies.record_cap(super::super::latency::ResponseCap::ResultTokens,"recent");}
            Ok(result)
        });
        let result =
            tokio::time::timeout_at(tokio::time::Instant::from_std(owner.deadline), operation)
                .await
                .unwrap_or_else(|_| Err(Error::Search(super::super::Error::Deadline)));
        self.observe_response_caps("recent", &result);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relative_and_iso_dates_are_checked_without_loose_git_date_aliases() {
        assert_eq!(since(Some("7d"), 1_000_000).unwrap(), 395200);
        assert_eq!(since(Some("2026-01-01"), 0).unwrap(), 1767225600);
        for invalid in ["yesterday", "-1d", "99999999999999999999d", "2026-99-01"] {
            assert!(since(Some(invalid), 0).is_err());
        }
    }
}
