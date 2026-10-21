//! Verified text prompt inputs and transitive provenance; v1 §5 and §6.8.
use super::{Error, Kind, source};
use crate::{
    config::Config,
    store::enrichment::publication::{Owner, Source},
    treat::prompts,
};
use rusqlite::{OptionalExtension, Transaction, params};
use std::{collections::BTreeMap, path::Path};

/// Prepared text and its complete input identity; never log source-bearing fields.
#[derive(Clone)]
pub(crate) struct Text {
    pub(crate) owner: Owner,
    pub(crate) parent_unit: i64,
    pub(crate) parent_hash: [u8; 32],
    pub(crate) fingerprint: [u8; 32],
    pub(crate) prompt: String,
    pub(crate) sources: Vec<Source>,
}
/// Reuses symbol and supplementary-unit proofs within one module proof pass.
/// Construct a fresh instance at every publication/serving freshness boundary.
#[derive(Default)]
pub(super) struct SymbolInputs {
    symbols: BTreeMap<i64, Option<Text>>,
    contexts: BTreeMap<i64, Option<source::FileInput>>,
}

// Keep text jobs inside the smallest supported chat context. PR collation can
// contain a very large squash body, so it gets a bounded head/tail view while
// ranked immutable hunks retain their own budget. Stored collation and source
// proofs remain complete.
const PR_COLLATION_TOKENS: usize = 4_000;
const PR_HUNK_TOKENS: usize = 3_000;
const PR_PROMPT_TOKENS: usize = 7_500;
const PR_INPUT_OMISSION: &str = "\n[Further content omitted at PR prompt input limit.]\n";

fn part(hash: &mut blake3::Hasher, bytes: &[u8]) {
    hash.update(&(bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}
fn evidence(hash: &mut blake3::Hasher, input: &source::FileInput) {
    part(hash, &input.occurrence.id.to_le_bytes());
    part(hash, &input.occurrence.content_hash);
    part(hash, &input.policy.fingerprint);
}

/// Captures a symbol's signature, verified L1 code, at most ten caller names and
/// two mentioning doc sections. Protected/unverifiable supplementary inputs are
/// omitted before rendering; the full preparation is repeated at worker fences.
pub(crate) fn symbol(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    id: i64,
) -> Result<Option<Text>, Error> {
    symbol_with_work(
        tx,
        root,
        config_root,
        config,
        id,
        crate::sync::l1::Work::default(),
    )
}

/// Prepares verified text while honoring the caller's cancellation and deadline.
pub(crate) fn symbol_with_work(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    id: i64,
    work: crate::sync::l1::Work<'_>,
) -> Result<Option<Text>, Error> {
    Ok(symbol_with_parent(tx, root, config_root, config, id, work)?.map(|(input, _)| input))
}

/// Keeps the verified parent used by this prompt so summary resolution does not
/// rebuild it. Publication and serving still repeat the full preparation.
pub(super) fn symbol_with_parent(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    id: i64,
    work: crate::sync::l1::Work<'_>,
) -> Result<Option<(Text, source::FileInput)>, Error> {
    symbol_with_contexts(
        tx,
        root,
        config_root,
        config,
        id,
        work,
        &mut BTreeMap::new(),
    )
}

/// Prepares one symbol using supplementary proofs owned by this source pass.
pub(super) fn symbol_with_contexts(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    id: i64,
    work: crate::sync::l1::Work<'_>,
    contexts: &mut BTreeMap<i64, Option<source::FileInput>>,
) -> Result<Option<(Text, source::FileInput)>, Error> {
    source::check_work(work)?;
    let target: Option<(i64, String, String, String)> = tx
        .query_row(
            "SELECT s.unit_id,s.name,coalesce(s.qualified,s.name),coalesce(s.signature,u.header)
         FROM symbols s JOIN units u ON u.id=s.unit_id WHERE s.id=?1 AND u.file_id=s.file_id",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .optional()?;
    let Some((parent_unit, name, qualified, signature)) = target else {
        return Ok(None);
    };
    let Some(parent) =
        source::resolve_enrichment_with_work(tx, root, config_root, config, parent_unit, work)?
    else {
        return Ok(None);
    };
    if parent.policy.never_send {
        return Ok(None);
    }
    let mut hash = blake3::Hasher::new_derive_key("dkc symbol enrichment input v1");
    part(&mut hash, &id.to_le_bytes());
    evidence(&mut hash, &parent);
    let mut sources = vec![Source::Symbol(id)];
    let mut callers = Vec::new();
    let mut query = tx.prepare(
        "SELECT DISTINCT s.id,coalesce(s.qualified,s.name),s.unit_id,s.pagerank
        FROM edges e JOIN symbols s ON s.id=e.src_symbol WHERE e.kind IN ('ref','test_of')
        AND (e.dst_symbol=?1 OR (e.dst_symbol IS NULL AND e.dst_name IN (?2,?3)))
        AND s.unit_id IS NOT NULL ORDER BY s.pagerank DESC,s.id LIMIT 10",
    )?;
    let rows = query.query_map(params![id, name, qualified], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
        ))
    })?;
    for row in rows {
        source::check_work(work)?;
        let (caller, name, unit) = row?;
        let input = match contexts.entry(unit) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => entry.insert(
                source::resolve_enrichment_context(tx, root, config_root, config, unit, work)?
                    .map(|input| input.file),
            ),
        };
        let Some(input) = input else {
            continue;
        };
        if input.policy.never_send {
            continue;
        }
        evidence(&mut hash, input);
        part(&mut hash, &caller.to_le_bytes());
        sources.extend([Source::Symbol(caller), Source::Unit(unit)]);
        callers.push(name);
    }
    let mut docs = Vec::new();
    let mut query=tx.prepare("SELECT DISTINCT u.id FROM edges e JOIN symbols s ON s.id=e.src_symbol
        JOIN units u ON u.id=s.unit_id WHERE e.kind='doc_mention' AND u.kind='doc_section'
        AND (e.dst_symbol=?1 OR (e.dst_symbol IS NULL AND e.dst_name IN (?2,?3))) ORDER BY u.id LIMIT 2")?;
    let rows = query.query_map(params![id, name, qualified], |r| r.get::<_, i64>(0))?;
    for row in rows {
        source::check_work(work)?;
        let unit = row?;
        let input = match contexts.entry(unit) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => entry.insert(
                source::resolve_enrichment_context(tx, root, config_root, config, unit, work)?
                    .map(|input| input.file),
            ),
        };
        let Some(input) = input else {
            continue;
        };
        if input.policy.never_send {
            continue;
        }
        evidence(&mut hash, input);
        sources.push(Source::Unit(unit));
        docs.push(input.context.text.clone());
    }
    let template = prompts::Template::load(
        config_root,
        &config.enrich.prompts_dir,
        Kind::DescribeSymbol,
    )
    .map_err(|_| Error::Invalid("enrichment prompt override"))?;
    let callers = callers.join("\n");
    let docs = docs.join("\n\n");
    let fields = BTreeMap::from([
        ("signature", signature.as_str()),
        ("code", parent.context.text.as_str()),
        ("callers", callers.as_str()),
        ("docs", docs.as_str()),
    ]);
    let prompt = template
        .render(&fields, 256 * 1024)
        .map_err(|_| Error::Invalid("enrichment prompt capacity or fields"))?;
    part(&mut hash, &template.fingerprint);
    part(&mut hash, prompt.as_bytes());
    Ok(Some((
        Text {
            owner: Owner::Symbol(id),
            parent_unit,
            parent_hash: parent.occurrence.content_hash,
            fingerprint: *hash.finalize().as_bytes(),
            prompt,
            sources,
        },
        parent,
    )))
}

/// Captures an owned PR collation and bounded immutable member hunks. Members
/// with higher-ranked changed files come first, then files by rank and path.
pub(crate) fn pr(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    id: i64,
) -> Result<Option<Text>, Error> {
    pr_with_work(
        tx,
        root,
        config_root,
        config,
        id,
        crate::sync::l1::Work::default(),
    )
}

/// Prepares verified text while honoring the caller's cancellation and deadline.
pub(crate) fn pr_with_work(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    id: i64,
    work: crate::sync::l1::Work<'_>,
) -> Result<Option<Text>, Error> {
    Ok(pr_with_parent(tx, root, config_root, config, id, work)?.map(|(input, _)| input))
}

/// Keeps the verified parent used by this prompt so summary resolution does not
/// rebuild it. Publication and serving still repeat the full preparation.
pub(super) fn pr_with_parent(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    id: i64,
    work: crate::sync::l1::Work<'_>,
) -> Result<Option<(Text, source::FileInput)>, Error> {
    source::check_work(work)?;
    let row: Option<(i64, i64)> = tx
        .query_row(
            "SELECT unit_id,commit_count FROM prs WHERE id=?1 AND unit_id IS NOT NULL",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    let Some((parent_unit, count)) = row else {
        return Ok(None);
    };
    if count < 1 || count as u64 > config.limits.max_pr_commits {
        return Ok(None);
    }
    let Some(parent) =
        source::resolve_enrichment_with_work(tx, root, config_root, config, parent_unit, work)?
    else {
        return Ok(None);
    };
    if parent.policy.never_send {
        return Ok(None);
    }
    let Some(mut members) =
        crate::store::Store::pr_members(tx, id, parent_unit, count as usize, &|| {
            work.check()
                .map_err(|_| crate::store::Error::L1InputsChanged)
        })?
    else {
        return Ok(None);
    };
    let mut ranks = BTreeMap::<String, f64>::new();
    for member in &members {
        source::check_work(work)?;
        for path in &member.files {
            source::check_work(work)?;
            if !ranks.contains_key(path) {
                let rank = tx
                    .query_row("SELECT rank FROM files WHERE path=?1", [path], |r| r.get(0))
                    .optional()?
                    .unwrap_or(0.0);
                ranks.insert(path.clone(), rank);
            }
        }
    }
    let rank = |member: &crate::store::prs::Member| {
        member
            .files
            .iter()
            .filter_map(|p| ranks.get(p))
            .copied()
            .fold(0.0_f64, f64::max)
    };
    members.sort_by(|a, b| {
        rank(b)
            .total_cmp(&rank(a))
            .then_with(|| b.time.cmp(&a.time))
            .then_with(|| a.sha.cmp(&b.sha))
    });
    let repo = gix::open(root).map_err(|_| Error::Invalid("PR hunk repository"))?;
    let file_bytes = config
        .limits
        .max_file_bytes()
        .map_err(|_| Error::Invalid("PR hunk source cap"))?
        .0;
    let mut hunks = String::new();
    let mut sources = Vec::new();
    let mut hash = blake3::Hasher::new_derive_key("dkc PR enrichment input v1");
    part(&mut hash, &id.to_le_bytes());
    evidence(&mut hash, &parent);
    for mut member in members {
        source::check_work(work)?;
        sources.push(Source::Commit(member.id));
        part(&mut hash, &member.sha);
        let remaining = PR_HUNK_TOKENS.saturating_sub(crate::treat::count_tokens(&hunks));
        if remaining <= 8 {
            continue;
        }
        member
            .files
            .sort_by(|a, b| ranks[b].total_cmp(&ranks[a]).then_with(|| a.cmp(b)));
        let oid = gix::ObjectId::try_from(member.sha.as_slice())
            .map_err(|_| Error::Invalid("PR member object ID"))?;
        let selected = crate::sync::history::hunks::read(
            &repo,
            oid,
            &member.files,
            crate::sync::history::hunks::Limits {
                file_bytes,
                files: config.limits.max_files,
                tokens: remaining - 8,
            },
            work,
        )
        .map_err(|_| Error::Invalid("PR hunk preparation bound or source"))?;
        let previous = hunks.len();
        hunks.push_str(&selected.text);
        if crate::treat::count_tokens(&hunks) > PR_HUNK_TOKENS {
            hunks.truncate(previous);
        }
    }
    let template =
        prompts::Template::load(config_root, &config.enrich.prompts_dir, Kind::SummarizePr)
            .map_err(|_| Error::Invalid("PR prompt override"))?;
    let collation = bounded_pr_collation(&parent.context.text);
    let prompt = template
        .render(
            &BTreeMap::from([("collation", collation.as_ref()), ("hunks", hunks.as_str())]),
            256 * 1024,
        )
        .map_err(|_| Error::Invalid("PR prompt capacity or fields"))?;
    if crate::treat::count_tokens(&prompt) > PR_PROMPT_TOKENS {
        return Err(Error::Invalid("PR prompt token bound"));
    }
    part(&mut hash, &template.fingerprint);
    part(&mut hash, prompt.as_bytes());
    Ok(Some((
        Text {
            owner: Owner::Pr(id),
            parent_unit,
            parent_hash: parent.occurrence.content_hash,
            fingerprint: *hash.finalize().as_bytes(),
            prompt,
            sources,
        },
        parent,
    )))
}

/// Rebuilds the exact text-job generation without provider credentials or network resources.
pub(crate) fn desired(config: &Config, kind: Kind, input: &Text) -> Result<[u8; 32], Error> {
    let maximum = prompts::output_tokens(kind)
        .map_err(|_| Error::Invalid("enrichment output ceiling"))?
        .ok_or(Error::Invalid("text output ceiling"))?;
    let wire = crate::models::openai_compat::chat::text_fingerprint(
        &config.llm,
        "Use only the supplied repository evidence.",
        &input.prompt,
        maximum,
    )
    .map_err(|_| Error::Invalid("enrichment chat preparation"))?;
    let endpoint = crate::models::openai_compat::chat::endpoint_identity(&config.llm.url)
        .map_err(|_| Error::Invalid("enrichment endpoint identity"))?;
    Ok(generation(&input.fingerprint, &wire, &endpoint))
}

/// Combines exact request bytes with source proof and provider destination.
pub(super) fn generation(input: &[u8; 32], wire: &[u8; 32], endpoint: &str) -> [u8; 32] {
    let mut hash = blake3::Hasher::new_derive_key("dkc enrichment desired request v1");
    hash.update(input);
    hash.update(wire);
    part(&mut hash, endpoint.as_bytes());
    *hash.finalize().as_bytes()
}

// Reserve separate token budgets for canonical collation and ranked member
// descriptions.  The 5,200-token input ceiling leaves room for the system
// message and the 200-token module response on the smallest supported chat
// context, while keeping aggregate repository evidence modest.
const MODULE_FIELD_TOKENS: usize = 2_500;
const MODULE_PROMPT_TOKENS: usize = 5_200;
const MODULE_INPUT_OMISSION: &str = "\n[Further content omitted at module prompt input limit.]\n";

fn prefix_at_most_tokens(text: &str, maximum: usize) -> &str {
    if crate::treat::count_tokens(text) <= maximum {
        return text;
    }
    let mut low = 0;
    let mut high = text.len();
    while low < high {
        let mut middle = low + (high - low).div_ceil(2);
        while middle > low && !text.is_char_boundary(middle) {
            middle -= 1;
        }
        if middle == low {
            break;
        }
        if crate::treat::count_tokens(&text[..middle]) <= maximum {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    &text[..low]
}

fn suffix_at_most_tokens(text: &str, maximum: usize) -> &str {
    if crate::treat::count_tokens(text) <= maximum {
        return text;
    }
    let mut low = 0;
    let mut high = text.len();
    while low < high {
        let mut middle = low + (high - low) / 2;
        while middle < high && !text.is_char_boundary(middle) {
            middle += 1;
        }
        if middle == high {
            break;
        }
        if crate::treat::count_tokens(&text[middle..]) <= maximum {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    while low < text.len() && !text.is_char_boundary(low) {
        low += 1;
    }
    &text[low..]
}

/// Keeps the title/overview and trailing details of a large PR collation.
fn bounded_pr_collation(text: &str) -> std::borrow::Cow<'_, str> {
    if crate::treat::count_tokens(text) <= PR_COLLATION_TOKENS {
        return std::borrow::Cow::Borrowed(text);
    }
    let marker = crate::treat::count_tokens(PR_INPUT_OMISSION);
    let first = (PR_COLLATION_TOKENS.saturating_sub(marker)) / 2;
    let last = PR_COLLATION_TOKENS.saturating_sub(marker + first);
    let bounded = format!(
        "{}{}{}",
        prefix_at_most_tokens(text, first),
        PR_INPUT_OMISSION,
        suffix_at_most_tokens(text, last),
    );
    debug_assert!(crate::treat::count_tokens(&bounded) <= PR_COLLATION_TOKENS);
    std::borrow::Cow::Owned(bounded)
}

/// Keeps both the overview and trailing exports/dependencies of large modules.
/// Only the outbound prompt is shortened; canonical content and source proof stay whole.
fn bounded_module_collation(text: &str) -> std::borrow::Cow<'_, str> {
    if crate::treat::count_tokens(text) <= MODULE_FIELD_TOKENS {
        return std::borrow::Cow::Borrowed(text);
    }
    let marker = crate::treat::count_tokens(MODULE_INPUT_OMISSION);
    let first = (MODULE_FIELD_TOKENS.saturating_sub(marker)) / 2;
    let last = MODULE_FIELD_TOKENS.saturating_sub(marker + first);
    let bounded = format!(
        "{}{}{}",
        prefix_at_most_tokens(text, first),
        MODULE_INPUT_OMISSION,
        suffix_at_most_tokens(text, last),
    );
    debug_assert!(crate::treat::count_tokens(&bounded) <= MODULE_FIELD_TOKENS);
    std::borrow::Cow::Owned(bounded)
}

/// Uses current member descriptions only after rebuilding each symbol's full generation.
pub(crate) fn module(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
) -> Result<Option<Text>, Error> {
    module_with_work(
        tx,
        root,
        config_root,
        config,
        unit,
        crate::sync::l1::Work::default(),
    )
}

/// Prepares verified text while honoring the caller's cancellation and deadline.
pub(crate) fn module_with_work(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
    work: crate::sync::l1::Work<'_>,
) -> Result<Option<Text>, Error> {
    Ok(module_with_parent(tx, root, config_root, config, unit, work)?.map(|(input, _)| input))
}

/// Returns the module prompt with its final verified parent observation. Keeping
/// both within this preparation avoids a second full source census in summary
/// resolution; callers must still repeat preparation at their publication fence.
pub(super) fn module_with_parent(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
    work: crate::sync::l1::Work<'_>,
) -> Result<Option<(Text, source::FileInput)>, Error> {
    let mut symbols = SymbolInputs::default();
    module_with_parent_with_symbols_and_parent(
        tx,
        root,
        config_root,
        config,
        unit,
        work,
        &mut symbols,
        None,
    )
}

/// Shares verified symbol prompt preparations only within one enclosing source
/// proof. Each module still rebuilds its own collation, parent observation and
/// generation identity before it can authorize a stored summary.
pub(super) fn module_with_parent_with_symbols(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
    work: crate::sync::l1::Work<'_>,
    symbols: &mut SymbolInputs,
) -> Result<Option<(Text, source::FileInput)>, Error> {
    module_with_parent_with_symbols_and_parent(
        tx,
        root,
        config_root,
        config,
        unit,
        work,
        symbols,
        None,
    )
}

/// Accepts one same-pass batch observation for the module parent. Callers must
/// supply the exact enrichment-level proof; absent evidence rebuilds it locally.
#[allow(
    clippy::too_many_arguments,
    reason = "explicit source proof inputs keep the cache boundary visible"
)]
pub(super) fn module_with_parent_with_symbols_and_parent(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
    work: crate::sync::l1::Work<'_>,
    symbols: &mut SymbolInputs,
    observed_parent: Option<source::FileInput>,
) -> Result<Option<(Text, source::FileInput)>, Error> {
    source::check_work(work)?;
    use crate::{
        store::enrichment::publication::{self, Owner, Source},
        treat::prompts,
    };
    let path: Option<String> = tx
        .query_row(
            "SELECT path FROM dirs WHERE module_unit=?1 AND file_count>=?2",
            params![
                unit,
                config.enrich.modules_min_files.min(i64::MAX as u64) as i64
            ],
            |r| r.get(0),
        )
        .optional()?;
    let Some(path) = path else {
        return Ok(None);
    };
    let mut child_generations = Vec::new();
    let mut text = String::new();
    let mut sources = Vec::new();
    let mut query=tx.prepare("SELECT s.id,coalesce(s.qualified,s.name),s.summary_unit,u.parent_unit,u.content_hash,c.text
            FROM symbols s JOIN files f ON f.id=s.file_id JOIN units u ON u.id=s.summary_unit
            JOIN unit_contents c ON c.hash=u.content_hash
            WHERE (?1='' OR substr(f.path,1,length(?1)+1)=?1||'/') AND u.kind='summary'
            AND u.source_key='summary:symbol:'||s.id ORDER BY s.pagerank DESC,s.id")?;
    let rows = query.query_map([&path], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
            r.get::<_, i64>(3)?,
            r.get::<_, Vec<u8>>(4)?,
            r.get::<_, String>(5)?,
        ))
    })?;
    for row in rows {
        source::check_work(work)?;
        let (id, name, summary, parent_id, stored, body) = row?;
        let prepared = match symbols.symbols.get(&id) {
            Some(prepared) => prepared.clone(),
            None => {
                let prepared = symbol_with_contexts(
                    tx,
                    root,
                    config_root,
                    config,
                    id,
                    work,
                    &mut symbols.contexts,
                )?
                .map(|(input, _)| input);
                symbols.symbols.insert(id, prepared.clone());
                prepared
            }
        };
        let Some(prepared) = prepared else {
            continue;
        };
        let desired = desired(config, Kind::DescribeSymbol, &prepared)?;
        if prepared.parent_unit != parent_id
            || stored.as_slice()
                != publication::content_hash(&prepared.parent_hash, &desired, &body)
        {
            continue;
        }
        let line = format!("{name}: {body}\n");
        let mut candidate = text.clone();
        candidate.push_str(&line);
        candidate.push_str(MODULE_INPUT_OMISSION);
        if crate::treat::count_tokens(&candidate) > MODULE_FIELD_TOKENS {
            text.push_str(MODULE_INPUT_OMISSION);
            break;
        }
        child_generations.push((summary, desired));
        text.push_str(&line);
        sources.extend([Source::Unit(summary), Source::Symbol(id)]);
    }
    // Verify the complete module after collecting symbol prompt evidence. This
    // observation supplies both canonical collation and the returned context.
    let parent = match observed_parent {
        Some(parent) => Some(parent),
        None => source::resolve_enrichment_with_work(tx, root, config_root, config, unit, work)?,
    };
    let Some(parent) = parent else {
        return Ok(None);
    };
    if parent.policy.never_send {
        return Ok(None);
    }
    let mut hash = blake3::Hasher::new_derive_key("dkc module enrichment input v1");
    hash.update(&unit.to_le_bytes());
    hash.update(&parent.occurrence.content_hash);
    hash.update(&parent.policy.fingerprint);
    for (summary, desired) in child_generations {
        hash.update(&summary.to_le_bytes());
        hash.update(&desired);
    }
    let template = prompts::Template::load(
        config_root,
        &config.enrich.prompts_dir,
        Kind::SummarizeModule,
    )
    .map_err(|_| Error::Invalid("module prompt override"))?;
    let collation = bounded_module_collation(&parent.context.text);
    let prompt = template
        .render(
            &std::collections::BTreeMap::from([
                ("collation", collation.as_ref()),
                ("summaries", text.as_str()),
            ]),
            256 * 1024,
        )
        .map_err(|_| Error::Invalid("module prompt capacity or fields"))?;
    if crate::treat::count_tokens(&prompt) > MODULE_PROMPT_TOKENS {
        return Err(Error::Invalid("module prompt token bound"));
    }
    hash.update(&template.fingerprint);
    hash.update(&(prompt.len() as u64).to_le_bytes());
    hash.update(prompt.as_bytes());
    Ok(Some((
        Text {
            owner: Owner::Module(unit),
            parent_unit: unit,
            parent_hash: parent.occurrence.content_hash,
            fingerprint: *hash.finalize().as_bytes(),
            prompt,
            sources,
        },
        parent,
    )))
}

#[cfg(test)]
mod module_prompt_tests {
    use super::*;

    #[test]
    fn large_module_collation_keeps_overview_and_exports_with_an_explicit_bound() {
        let input = format!(
            "README overview\n{}\nexports:\npublic_entry\ndepends on:\ncore",
            "文件🦀\n".repeat(40_000)
        );
        let shortened = bounded_module_collation(&input);
        assert!(crate::treat::count_tokens(&shortened) <= MODULE_FIELD_TOKENS);
        assert!(shortened.starts_with("README overview\n"));
        assert!(shortened.ends_with("exports:\npublic_entry\ndepends on:\ncore"));
        assert!(shortened.contains(MODULE_INPUT_OMISSION));
        assert_eq!(shortened, bounded_module_collation(&input));
        let root = tempfile::tempdir().unwrap();
        let template = crate::treat::prompts::Template::load(
            root.path(),
            ".deepkit/prompts",
            Kind::SummarizeModule,
        )
        .unwrap();
        let full_summaries = "member: responsibility\n".repeat(10_000);
        let summaries = bounded_module_collation(&full_summaries);
        let prompt = template
            .render(
                &BTreeMap::from([
                    ("collation", shortened.as_ref()),
                    ("summaries", summaries.as_ref()),
                ]),
                256 * 1024,
            )
            .unwrap();
        assert!(prompt.contains("public_entry"));
        assert!(crate::treat::count_tokens(&prompt) <= MODULE_PROMPT_TOKENS);
        assert!(prompt.contains(summaries.as_ref()));
    }

    #[test]
    fn ordinary_module_collation_is_unchanged_including_at_the_field_boundary() {
        let at_boundary = "x ".repeat(MODULE_FIELD_TOKENS - 1);
        assert_eq!(
            crate::treat::count_tokens(&at_boundary),
            MODULE_FIELD_TOKENS
        );
        for input in ["small module".to_owned(), at_boundary] {
            assert!(matches!(
                bounded_module_collation(&input),
                std::borrow::Cow::Borrowed(_)
            ));
            assert_eq!(bounded_module_collation(&input), input);
        }
    }
}
