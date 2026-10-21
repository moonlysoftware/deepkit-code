//! Serialized embedding-route changes and physical completion; v1 §7.3.
use super::{Error, Store, UnitKind, admission, embedding_bindings, vectors::vector_table_name};
use rusqlite::{OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// One destination per kind, with at most two configured physical tables.
pub(crate) type Routes = BTreeMap<String, String>;
/// Only affected kinds are retained as fallback sources; unchanged kinds may
/// continue using this table in the desired routing map. Known is a provenance
/// hint, not authority to dispatch without rereading the provider binding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Source {
    pub(crate) table: String,
    pub(crate) kinds: BTreeSet<String>,
    pub(crate) known: bool,
}
/// Durable routing generation. Empty previous means stable; a pending generation
/// can finish only after its old affected vector rows disappear through replacement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct State {
    pub(crate) generation: i64,
    pub(crate) desired: Routes,
    pub(crate) previous: Vec<Source>,
}
fn refused(reason: &'static str) -> Error {
    Error::EmbeddingTransition { reason }
}
fn kind_valid(kind: &str) -> bool {
    UnitKind::ALL.iter().any(|value| value.as_str() == kind)
}
fn validate_routes(routes: &Routes) -> Result<(), Error> {
    if routes.len() != UnitKind::ALL.len()
        || routes.keys().any(|kind| !kind_valid(kind))
        || routes.values().collect::<BTreeSet<_>>().len() > 2
    {
        return Err(refused(
            "routing must cover every kind with at most two destinations",
        ));
    }
    for table in routes.values() {
        vector_table_name(table)?;
    }
    Ok(())
}
fn validate_sources(sources: &[Source]) -> Result<(), Error> {
    if sources.len() > 2
        || sources
            .iter()
            .map(|s| &s.table)
            .collect::<BTreeSet<_>>()
            .len()
            != sources.len()
    {
        return Err(refused(
            "legacy source layout exceeds one bounded transition",
        ));
    }
    for source in sources {
        vector_table_name(&source.table)?;
        if source.kinds.is_empty() || source.kinds.iter().any(|kind| !kind_valid(kind)) {
            return Err(refused("invalid source kinds"));
        }
    }
    Ok(())
}
/// Reads validated routing metadata in the caller's snapshot; no vector table is
/// touched by this operation, including steady single-mode search.
pub(crate) fn read(tx: &Transaction<'_>) -> Result<Option<State>, Error> {
    let raw: Option<(i64, String, String)> = tx
        .query_row(
            "SELECT generation,desired,previous FROM embedding_transition WHERE singleton=1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    raw.map(|(generation, desired, previous)| {
        let desired =
            serde_json::from_str(&desired).map_err(|_| refused("invalid stored routing"))?;
        let previous: Vec<Source> =
            serde_json::from_str(&previous).map_err(|_| refused("invalid stored sources"))?;
        validate_routes(&desired)?;
        validate_sources(&previous)?;
        if generation <= 0 {
            return Err(refused("invalid generation"));
        }
        Ok(State {
            generation,
            desired,
            previous,
        })
    })
    .transpose()
}
fn table_exists(tx: &Transaction<'_>, table: &str) -> Result<bool, Error> {
    Ok(tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM embedding_tables WHERE slug=?1)",
        [table],
        |r| r.get(0),
    )?)
}
fn kinds_in(tx: &Transaction<'_>, table: &str) -> Result<BTreeSet<String>, Error> {
    if !table_exists(tx, table)? {
        return Ok(BTreeSet::new());
    }
    let name = vector_table_name(table)?;
    Ok(tx
        .prepare(&format!(
            "SELECT DISTINCT u.kind FROM {name} v JOIN units u ON u.id=v.unit_id"
        ))?
        .query_map([], |r| r.get(0))?
        .collect::<Result<_, _>>()?)
}
fn remaining(tx: &Transaction<'_>, sources: &[Source]) -> Result<bool, Error> {
    for source in sources {
        if !table_exists(tx, &source.table)? {
            continue;
        }
        let table = vector_table_name(&source.table)?;
        let kinds = serde_json::to_string(&source.kinds).map_err(|_| refused("source encoding"))?;
        let exists: bool=tx.query_row(&format!("SELECT EXISTS(SELECT 1 FROM {table} v JOIN units u ON u.id=v.unit_id WHERE u.kind IN (SELECT value FROM json_each(?1)))"),[kinds],|r|r.get(0))?;
        if exists {
            return Ok(true);
        }
    }
    Ok(false)
}
fn sources(
    tx: &Transaction<'_>,
    desired: &Routes,
    prior: Option<&Routes>,
) -> Result<Vec<Source>, Error> {
    let tables: BTreeSet<String> = if let Some(prior) = prior {
        prior.values().cloned().collect()
    } else {
        tx.prepare("SELECT slug FROM embedding_tables ORDER BY slug")?
            .query_map([], |r| r.get(0))?
            .collect::<Result<_, _>>()?
    };
    let mut sources = Vec::new();
    for table in tables {
        let kinds = kinds_in(tx, &table)?
            .into_iter()
            .filter(|kind| {
                desired.get(kind) != Some(&table)
                    && prior.is_none_or(|prior| prior.get(kind) == Some(&table))
            })
            .collect::<BTreeSet<_>>();
        if !kinds.is_empty() {
            sources.push(Source {
                known: embedding_bindings::read(tx, &table)?.is_some(),
                table,
                kinds,
            });
        }
    }
    validate_sources(&sources)?;
    Ok(sources)
}
fn plan(tx: &Transaction<'_>, desired: Routes) -> Result<(State, bool), Error> {
    validate_routes(&desired)?;
    let prior = read(tx)?;
    if let Some(prior) = &prior {
        if prior.desired == desired {
            return Ok((prior.clone(), false));
        }
        if remaining(tx, &prior.previous)? {
            return Err(refused("another routing change is still pending"));
        }
    }
    let previous = sources(tx, &desired, prior.as_ref().map(|s| &s.desired))?;
    let generation = prior
        .as_ref()
        .map_or(Some(1), |s| s.generation.checked_add(1))
        .ok_or_else(|| refused("generation overflow"))?;
    Ok((
        State {
            generation,
            desired,
            previous,
        },
        true,
    ))
}
fn write(tx: &Transaction<'_>, state: &State) -> Result<(), Error> {
    let desired = serde_json::to_string(&state.desired).map_err(|_| refused("routing encoding"))?;
    let previous =
        serde_json::to_string(&state.previous).map_err(|_| refused("source encoding"))?;
    tx.execute("INSERT INTO embedding_transition(singleton,generation,desired,previous) VALUES(1,?1,?2,?3) ON CONFLICT(singleton) DO UPDATE SET generation=excluded.generation,desired=excluded.desired,previous=excluded.previous",params![state.generation,desired,previous])?;
    Ok(())
}
impl Store {
    /// Serializes a complete desired routing map before target preparation. The
    /// resolver checks current configuration and resolves all destinations inside
    /// each snapshot; a changed plan is rechecked under admitted writer ownership.
    /// Same-plan resumes are read-only, even while a transition remains pending.
    pub(crate) fn prepare_transition<E: From<Error> + From<rusqlite::Error>>(
        &self,
        resolve: impl Fn(&Transaction<'_>) -> Result<Routes, E>,
    ) -> Result<State, E> {
        let (state, change) = self.read_snapshot(|tx| {
            resolve(tx).and_then(|desired| plan(tx, desired).map_err(E::from))
        })??;
        if !change {
            return Ok(state);
        }
        admission::transaction(self, admission::Kind::Growth, |tx| {
            let (state, change) = plan(&tx, resolve(&tx)?)?;
            if change {
                write(&tx, &state)?;
            }
            tx.commit().map_err(Error::from)?;
            Ok(state)
        })
    }
    /// Completes only the expected generation after every affected old row is gone.
    /// A later generation cannot be completed by a stale host. Queue counts are
    /// deliberately irrelevant; missing old tables were retired only when empty.
    pub(crate) fn finish_transition(&self, generation: i64) -> Result<State, Error> {
        let inspect = |tx: &Transaction<'_>| {
            let mut state = read(tx)?.ok_or_else(|| refused("missing routing generation"))?;
            if state.generation != generation {
                return Err(refused("routing generation superseded"));
            }
            let change = !state.previous.is_empty() && !remaining(tx, &state.previous)?;
            if change {
                state.previous.clear();
            }
            Ok((state, change))
        };
        let (state, change) = self.read_snapshot(inspect)??;
        if !change {
            return Ok(state);
        }
        admission::transaction(self, admission::Kind::Control, |tx| {
            let (state, change) = inspect(&tx)?;
            if change {
                write(&tx, &state)?;
            }
            tx.commit()?;
            Ok(state)
        })
    }
}
#[cfg(test)]
mod tests;
