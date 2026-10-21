//! Preserving Codex project configuration merge for repository-bound stdio setup.
use anyhow::{Result, anyhow, bail};
use std::path::Path;

/// Adds a stdio server with an explicit canonical repository working directory.
/// Existing bytes and comments are retained, and an identical entry is unchanged.
/// Unsupported closed tables and conflicting entries are refused without input values.
pub(in crate::cli) fn merge(existing: Option<&[u8]>, root: &Path) -> Result<Vec<u8>> {
    let root = std::fs::canonicalize(root)
        .map_err(|_| anyhow!("Could not resolve the repository directory for Codex setup"))?;
    let cwd = root
        .to_str()
        .ok_or_else(|| anyhow!("Codex setup requires a UTF-8 repository path"))?;
    let entry = toml::Table::from_iter([
        ("command".into(), toml::Value::String("dkc".into())),
        (
            "args".into(),
            toml::Value::Array(vec![
                toml::Value::String("serve".into()),
                toml::Value::String("--stdio".into()),
            ]),
        ),
        ("cwd".into(), toml::Value::String(cwd.into())),
    ]);
    let mut document = parse(existing.unwrap_or_default())?;
    let servers = document
        .entry("mcp_servers")
        .or_insert_with(|| toml::Value::Table(toml::Table::new()));
    let Some(servers) = servers.as_table_mut() else {
        bail!("Codex configuration mcp_servers must be a TOML table");
    };
    let entry = toml::Value::Table(entry);
    if let Some(current) = servers.get("deepkit-code") {
        if current != &entry {
            bail!(
                "Codex configuration has a conflicting deepkit-code entry; review .codex/config.toml and its command, args and cwd before retrying"
            );
        }
        return Ok(existing.unwrap_or_default().to_vec());
    }
    let rendered =
        toml::to_string(&entry).map_err(|_| anyhow!("Could not render the Codex MCP entry"))?;
    servers.insert("deepkit-code".into(), entry);
    let mut bytes = existing.unwrap_or_default().to_vec();
    if !bytes.is_empty() && !bytes.ends_with(b"\n") {
        bytes.push(b'\n');
    }
    bytes.extend_from_slice(b"\n[mcp_servers.deepkit-code]\n");
    bytes.extend_from_slice(rendered.as_bytes());
    // Inline tables cannot be extended, and unusual dotted table forms can also
    // reject an appended block. Never reserialize the user's whole configuration
    // or assume that a syntactically valid append preserved its meaning.
    if parse(&bytes).ok().as_ref() != Some(&document) {
        bail!(
            "Codex configuration cannot be extended without rewriting existing tables; convert mcp_servers to standard [mcp_servers.NAME] tables and retry"
        );
    }
    Ok(bytes)
}

/// Parses TOML without exposing source text, including other servers' credentials.
pub(in crate::cli) fn parse(bytes: &[u8]) -> Result<toml::Table> {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|source| toml::from_str(source).ok())
        .ok_or_else(|| {
            anyhow!("Codex configuration must be valid UTF-8 TOML without duplicate keys")
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_bound_stdio_entry_preserving_comments_values_and_original_bytes() {
        let root = tempfile::tempdir().unwrap();
        for original in [
            "",
            "# project preferences\nmodel = 'example'\n",
            "# keep formatting\r\n[mcp_servers.other]\r\ncommand = 'other' # keep comment",
            "[mcp_servers]\nother = { command = 'other' }\n[features]\nexample = true\n",
        ] {
            let merged = merge(Some(original.as_bytes()), root.path()).unwrap();
            assert!(merged.starts_with(original.as_bytes()));
            let mut after = parse(&merged).unwrap();
            let servers = after
                .get_mut("mcp_servers")
                .unwrap()
                .as_table_mut()
                .unwrap();
            let entry = servers.remove("deepkit-code").unwrap();
            assert_eq!(entry["command"].as_str(), Some("dkc"));
            assert_eq!(
                entry["args"].as_array().unwrap(),
                &[toml::Value::from("serve"), toml::Value::from("--stdio")]
            );
            assert_eq!(
                entry["cwd"].as_str(),
                root.path().canonicalize().unwrap().to_str()
            );
            let before = parse(original.as_bytes()).unwrap();
            if !before.contains_key("mcp_servers") {
                after.remove("mcp_servers");
            }
            assert_eq!(after, before);
            assert_eq!(merge(Some(&merged), root.path()).unwrap(), merged);
        }
    }

    #[test]
    fn separate_repositories_receive_distinct_working_directories() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let first = parse(&merge(None, first.path()).unwrap()).unwrap();
        let second = parse(&merge(None, second.path()).unwrap()).unwrap();
        assert_ne!(
            first["mcp_servers"]["deepkit-code"]["cwd"],
            second["mcp_servers"]["deepkit-code"]["cwd"]
        );
    }

    #[test]
    fn refuses_conflicts_malformed_and_closed_tables_without_input_values() {
        let root = tempfile::tempdir().unwrap();
        for original in [
            "private-placeholder = [",
            "mcp_servers = 'private-placeholder'",
            "mcp_servers = {}",
            "mcp_servers = { other = { command = 'private-placeholder' } }",
            "[mcp_servers.deepkit-code]\ncommand='private-placeholder'",
            "[mcp_servers]\n[mcp_servers]\nprivate-placeholder=true",
        ] {
            let error = merge(Some(original.as_bytes()), root.path()).unwrap_err();
            assert!(!format!("{error:#} {error:?}").contains("private-placeholder"));
            assert!(error.to_string().starts_with("Codex configuration"));
        }
        assert!(merge(Some(&[0xff]), root.path()).is_err());
        let first = merge(None, root.path()).unwrap();
        let other = tempfile::tempdir().unwrap();
        assert!(merge(Some(&first), other.path()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn path_aliases_are_canonicalized_and_toml_strings_are_escaped() {
        let root = tempfile::tempdir().unwrap();
        let repository = root
            .path()
            .join("repo with 'quotes' and \"double\" and \\slashes");
        std::fs::create_dir(&repository).unwrap();
        let alias = root.path().join("alias");
        std::os::unix::fs::symlink(&repository, &alias).unwrap();
        let merged = merge(None, &alias).unwrap();
        assert_eq!(
            parse(&merged).unwrap()["mcp_servers"]["deepkit-code"]["cwd"].as_str(),
            repository.canonicalize().unwrap().to_str()
        );
    }
}
