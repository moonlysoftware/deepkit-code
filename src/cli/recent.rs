//! CLI presentation for the shared recent-history service; v1.md §10.2.
use super::*;

#[derive(clap::Args)]
pub(super) struct Options {
    /// A repository path/directory or symbol name.
    #[arg(conflicts_with_all=["path","symbol"])]
    target: Option<String>,
    #[arg(long)]
    path: Option<String>,
    #[arg(long)]
    symbol: Option<String>,
    #[arg(long)]
    since: Option<String>,
    #[arg(long)]
    commits: bool,
    #[arg(short = 'k', long)]
    k: Option<usize>,
    #[arg(long)]
    json: bool,
}

/// Resolves the shorthand once, then uses the same filters and authority as MCP.
pub(super) fn run(mut options: Options) -> Result<()> {
    let cwd = current_dir()?;
    if let Some(target) = options.target {
        if cwd.join(&target).exists() || target.contains('/') || target.contains('.') {
            options.path = Some(target);
        } else {
            options.symbol = Some(target);
        }
    }
    let context = search::service::Context::load(&cwd)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let response = runtime.block_on(context.recent(search::service::recent::Request {
        path: options.path,
        symbol: options.symbol,
        since: options.since,
        granularity: options.commits.then(|| "commit".into()),
        k: options.k,
    }));
    let response = match response {
        Ok(value) => value,
        Err(error) => {
            if options.json {
                outln!(
                    "{}",
                    json!({"error":{"code":error.code(),"message":error.message(),"hint":error.hint()}})
                );
            }
            return Err(error.into());
        }
    };
    if options.json {
        outln!("{}", timed_response(response));
    } else if let Some(rows) = response.as_array() {
        for row in rows {
            outln!(
                "{} {} {} {}",
                row["id"].as_str().unwrap_or(""),
                row["date"].as_str().unwrap_or(""),
                row["kind"].as_str().unwrap_or(""),
                row["title"].as_str().unwrap_or("")
            );
        }
    }
    Ok(())
}
