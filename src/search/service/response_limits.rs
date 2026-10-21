//! Whole-result response admission, excluding the fixed control envelope; v1.md §3.5.
use super::*;

/// Empty result containers carry zero content tokens; protocol metadata remains available.
#[derive(Debug, serde::Serialize)]
pub(super) struct Report {
    limit: u64,
    used_tokens: u64,
    considered: usize,
    returned: usize,
    pub(super) truncated: bool,
}
fn cost(value: &Value, work: sync::l1::Work<'_>) -> Result<u64> {
    check_query_work(work)?;
    let result = crate::treat::count_tokens(&value.to_string()) as u64;
    check_query_work(work)?;
    Ok(result)
}
/// Admits one complete item while collecting a source-verified prefix, so callers
/// need not inspect trailing sources that cannot contribute to the response.
pub(super) fn push(
    rows: &mut Vec<Value>,
    item: Value,
    maximum: u64,
    work: sync::l1::Work<'_>,
) -> Result<bool> {
    rows.push(item);
    if cost(&json!(rows), work)? > maximum {
        rows.pop();
        Ok(false)
    } else {
        Ok(true)
    }
}

/// Retains a measured feasible rank prefix in O(log n) tokenizations. No assumption
/// that BPE counts add across JSON boundaries is needed; result objects stay whole.
pub(super) fn array(
    rows: &mut Vec<Value>,
    maximum: u64,
    work: sync::l1::Work<'_>,
) -> Result<Report> {
    let considered = rows.len();
    let mut used = if rows.is_empty() {
        0
    } else {
        cost(&json!(rows), work)?
    };
    if used > maximum {
        let mut low = 0;
        let mut high = rows.len();
        used = 0;
        while high - low > 1 {
            let middle = low + (high - low) / 2;
            let trial = cost(&json!(&rows[..middle]), work)?;
            if trial <= maximum {
                low = middle;
                used = trial;
            } else {
                high = middle;
            }
        }
        rows.truncate(low);
    }
    check_query_work(work)?;
    Ok(Report {
        limit: maximum,
        used_tokens: used,
        considered,
        returned: rows.len(),
        truncated: rows.len() < considered,
    })
}
/// Fits snippets and optional summary previews before dropping whole hits, preserving the original rank
/// prefix and all identity, score and freshness fields. Full text remains in open.
pub(super) fn search(
    rows: &mut Vec<Value>,
    maximum: u64,
    work: sync::l1::Work<'_>,
) -> Result<Report> {
    if rows.is_empty() || cost(&json!(rows), work)? <= maximum {
        return array(rows, maximum, work);
    }
    let originals = std::mem::take(rows);
    let considered = originals.len();
    let mut metadata = originals.clone();
    for row in &mut metadata {
        if row.get("snippet").is_some_and(Value::is_string) {
            row["snippet"] = json!("");
        }
        if let Some(object) = row.as_object_mut() {
            object.remove("summary");
        }
    }
    let mut report = array(&mut metadata, maximum, work)?;
    let count = metadata.len();
    let retained = &originals[..count];
    let full_cost = if count == 0 {
        0
    } else {
        cost(&json!(retained), work)?
    };
    if full_cost <= maximum {
        *rows = retained.to_vec();
        report.used_tokens = full_cost;
    } else {
        // Scale every preview equally. Retain only measured feasible candidates;
        // BPE boundary changes need not make token counts monotonic.
        let mut low = 0;
        let mut high = 1000;
        *rows = metadata;
        while high - low > 1 {
            let middle = low + (high - low) / 2;
            let mut trial = retained.to_vec();
            for row in &mut trial {
                for field in ["snippet", "summary"] {
                    if let Some(text) = row.get(field).and_then(Value::as_str) {
                        let keep = text.chars().count() * middle / 1000;
                        let prefix = text.chars().take(keep).collect::<String>();
                        if field == "summary" && keep == 0 {
                            if let Some(object) = row.as_object_mut() {
                                object.remove(field);
                            }
                            continue;
                        }
                        row[field] = json!(if keep == 0 {
                            prefix
                        } else {
                            format!("{prefix}…")
                        });
                    }
                }
            }
            let used = cost(&json!(trial), work)?;
            if used <= maximum {
                low = middle;
                *rows = trial;
                report.used_tokens = used;
            } else {
                high = middle;
            }
        }
    }
    report.considered = considered;
    report.returned = rows.len();
    report.truncated = rows.len() < considered;
    Ok(report)
}

/// Keeps map tree text and structured nodes synchronized while applying the shared
/// response cap in addition to the caller's tree budget. Complete file nodes survive.
pub(super) fn map(
    mapped: &mut super::super::map::MapResult,
    maximum: u64,
    tree_budget: usize,
    work: sync::l1::Work<'_>,
) -> Result<Report> {
    let considered = mapped.nodes.len();
    let mut used = if considered == 0 {
        0
    } else {
        cost(&serde_json::to_value(&*mapped)?, work)?
    };
    if used > maximum {
        let original = mapped.nodes.clone();
        let mut low = 0;
        let mut high = original.len();
        let mut best_nodes = Vec::new();
        let mut best_tree = String::new();
        used = 0;
        while high - low > 1 {
            let middle = low + (high - low) / 2;
            mapped.nodes = original[..middle].to_vec();
            let paths = mapped.nodes.iter().map(|n| n.path.clone()).collect();
            super::super::map::retain_checked(mapped, &paths, tree_budget, &|| {
                work.check().map_err(|_| store::Error::L1InputsChanged)
            })?;
            let trial = if mapped.nodes.is_empty() {
                0
            } else {
                cost(&serde_json::to_value(&*mapped)?, work)?
            };
            if trial <= maximum {
                low = middle;
                used = trial;
                best_nodes = mapped.nodes.clone();
                best_tree = mapped.tree.clone();
            } else {
                high = middle;
            }
        }
        mapped.nodes = best_nodes;
        mapped.tree = best_tree;
    }
    check_query_work(work)?;
    Ok(Report {
        limit: maximum,
        used_tokens: used,
        considered,
        returned: mapped.nodes.len(),
        truncated: mapped.nodes.len() < considered,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whole_rank_prefix_handles_exact_boundaries_zero_and_an_oversized_first_hit() -> Result<()> {
        let original = vec![
            json!({"id":"α","text":"int first;"}),
            json!({"id":"β","text":"int second;"}),
            json!({"id":"γ","text":"third"}),
        ];
        let one = crate::treat::count_tokens(&json!(&original[..1]).to_string()) as u64;
        for maximum in [0, one - 1, one, one + 1, 10000] {
            let mut rows = original.clone();
            let report = array(&mut rows, maximum, Default::default())?;
            assert_eq!(rows, original[..rows.len()]);
            assert!(report.used_tokens <= maximum);
            assert_eq!(report.truncated, rows.len() < original.len());
            if maximum < one {
                assert!(rows.is_empty());
            }
            if maximum == one {
                assert_eq!(rows.len(), 1);
            }
        }
        let mut rows = vec![
            json!({"text":"large ".repeat(4096)}),
            json!({"text":"small"}),
        ];
        array(&mut rows, 10, Default::default())?;
        assert!(rows.is_empty());
        Ok(())
    }
}

#[cfg(test)]
mod preview_tests {
    use super::*;

    #[test]
    fn search_shares_preview_space_before_dropping_ranked_hits() {
        let work = sync::l1::Work::default();
        let text = "日本語🙂 long source line ".repeat(20);
        let original = vec![
            json!({"id":"u_first", "score":0.9, "fresh":{"state":"fresh"}, "snippet":text}),
            json!({"id":"u_second", "score":0.8, "fresh":{"state":"fresh"}, "snippet":text}),
        ];
        let mut metadata = original.clone();
        for row in &mut metadata {
            row["snippet"] = json!("");
        }
        let cap = cost(&json!(metadata), work).unwrap() + 20;
        let mut rows = original.clone();
        let report = search(&mut rows, cap, work).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(!report.truncated);
        assert_eq!(report.considered, 2);
        assert_eq!(report.used_tokens, cost(&json!(rows), work).unwrap());
        assert!(report.used_tokens <= cap);
        for (actual, expected) in rows.iter().zip(&original) {
            assert_eq!(actual["id"], expected["id"]);
            assert_eq!(actual["score"], expected["score"]);
            assert_eq!(actual["fresh"], expected["fresh"]);
            let preview = actual["snippet"].as_str().unwrap();
            assert!(text.starts_with(preview.strip_suffix('…').unwrap_or(preview)));
        }
        let mut none = original.clone();
        let zero = search(&mut none, 0, work).unwrap();
        assert!(none.is_empty());
        assert!(zero.truncated);
        let mut full = original.clone();
        search(&mut full, 10000, work).unwrap();
        assert_eq!(full, original);
    }

    #[test]
    fn optional_summaries_share_preview_budget_without_displacing_ranked_hits() {
        let work = sync::l1::Work::default();
        let text = "日本語🙂 verified explanation ".repeat(100);
        let original = vec![
            json!({"id":"u_first", "score":0.9, "snippet":text, "summary":text}),
            json!({"id":"u_second", "score":0.8, "snippet":text, "summary":text}),
        ];
        let floor = json!([
            {"id":"u_first", "score":0.9, "snippet":""},
            {"id":"u_second", "score":0.8, "snippet":""}
        ]);
        for cap in [
            cost(&floor, work).unwrap(),
            cost(&floor, work).unwrap() + 80,
        ] {
            let mut rows = original.clone();
            let report = search(&mut rows, cap, work).unwrap();
            assert_eq!(rows.len(), 2);
            assert!(!report.truncated);
            assert_eq!(report.used_tokens, cost(&json!(rows), work).unwrap());
            assert!(report.used_tokens <= cap);
            for (actual, expected) in rows.iter().zip(&original) {
                assert_eq!(actual["id"], expected["id"]);
                assert_eq!(actual["score"], expected["score"]);
                for field in ["snippet", "summary"] {
                    if let Some(preview) = actual[field].as_str() {
                        assert!(text.starts_with(preview.strip_suffix('…').unwrap_or(preview)));
                    }
                }
            }
        }
    }
}
