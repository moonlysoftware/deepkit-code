//! Pure, preserving project MCP configuration merge for init; v1 §10.2.
use anyhow::{Result, bail};
use serde::{Deserialize, Deserializer, de};
use serde_json::{Map, Value};
use std::fmt;

/// Adds the specified stdio entry while preserving unrelated JSON values.
/// An identical existing entry preserves every original byte; a conflicting
/// entry is refused. The caller owns bounded reading and guarded publication.
/// Errors never include input values, which may contain another server's secrets.
pub(super) fn merge_mcp(existing: Option<&[u8]>) -> Result<Vec<u8>> {
    let mut document = match existing {
        Some(bytes) => parse_mcp(bytes)?,
        None => Value::Object(Map::new()),
    };
    let Some(root) = document.as_object_mut() else {
        bail!("MCP configuration must be a JSON object");
    };
    let servers = root
        .entry("mcpServers")
        .or_insert_with(|| Value::Object(Map::new()));
    let Some(servers) = servers.as_object_mut() else {
        bail!("MCP configuration mcpServers must be a JSON object");
    };
    let entry = serde_json::json!({"command": "dkc", "args": ["serve", "--stdio"]});
    if let Some(current) = servers.get("deepkit-code") {
        if current != &entry {
            bail!("MCP configuration has a conflicting deepkit-code entry");
        }
        if let Some(bytes) = existing {
            return Ok(bytes.to_vec());
        }
    }
    servers.insert("deepkit-code".into(), entry);
    let mut bytes = serde_json::to_vec_pretty(&document)
        .map_err(|_| anyhow::anyhow!("Could not render MCP configuration"))?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Parses JSON without accepting duplicate keys, retaining no input in errors.
/// Shared by init publication and read-only doctor checks so ambiguity is consistent.
pub(in crate::cli) fn parse_mcp(bytes: &[u8]) -> Result<Value> {
    serde_json::from_slice::<UniqueValue>(bytes)
        .map(|value| value.0)
        .map_err(|_| anyhow::anyhow!("MCP configuration must be valid JSON without duplicate keys"))
}

// serde_json::Value normally accepts duplicate keys by keeping only the last
// value. Reject them at every depth so merging cannot silently lose user data.
struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> de::Visitor<'de> for Visitor {
            type Value = UniqueValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value without duplicate keys")
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Bool(value)))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Number(value.into())))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Number(value.into())))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(|number| UniqueValue(Value::Number(number)))
                    .ok_or_else(|| E::custom("invalid JSON number"))
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::String(value.into())))
            }

            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::String(value)))
            }

            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Null))
            }

            fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element::<UniqueValue>()? {
                    values.push(value.0);
                }
                Ok(UniqueValue(Value::Array(values)))
            }

            fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(de::Error::custom("duplicate JSON key"));
                    }
                    values.insert(key, map.next_value::<UniqueValue>()?.0);
                }
                Ok(UniqueValue(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_exact_entry_and_preserves_unrelated_values() {
        let fresh = merge_mcp(None).unwrap();
        let entry = serde_json::json!({"command":"dkc","args":["serve","--stdio"]});
        assert_eq!(
            serde_json::from_slice::<Value>(&fresh).unwrap(),
            serde_json::json!({"mcpServers":{"deepkit-code":entry}})
        );
        assert!(fresh.ends_with(b"\n"));
        let original = br#"{"metadata":{"enabled":true,"unset":null,"counts":[-1,18446744073709551615,0.125]},"mcpServers":{"other":{"command":"other","args":["--option"],"env":{"VARIABLE":"placeholder"}}}}"#;
        let before: Value = serde_json::from_slice(original).unwrap();
        let merged = merge_mcp(Some(original)).unwrap();
        let after: Value = serde_json::from_slice(&merged).unwrap();
        assert_eq!(after["metadata"], before["metadata"]);
        assert_eq!(after["mcpServers"]["other"], before["mcpServers"]["other"]);
        assert_eq!(after["mcpServers"]["deepkit-code"], entry);
        assert_eq!(merge_mcp(Some(&merged)).unwrap(), merged);
        let absent = merge_mcp(Some(br#"{"metadata":7}"#)).unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&absent).unwrap()["metadata"],
            7
        );
    }

    #[test]
    fn identical_entry_keeps_original_whitespace_and_order() {
        let original = b" {\r\n \"mcpServers\": { \"deepkit-code\": {\"args\":[\"serve\",\"--stdio\"], \"command\":\"dkc\"} }, \"extra\": 1 } \t";
        assert_eq!(merge_mcp(Some(original)).unwrap(), original);
    }

    #[test]
    fn refuses_conflicts_without_reporting_values() {
        for entry in [
            r#"{"command":"private-placeholder","args":["serve","--stdio"]}"#,
            r#"{"command":"dkc","args":["serve","--stdio"],"env":{}}"#,
            "null",
        ] {
            let original = format!(r#"{{"mcpServers":{{"deepkit-code":{entry}}}}}"#);
            let error = merge_mcp(Some(original.as_bytes())).unwrap_err();
            assert_eq!(
                format!("{error:#}"),
                "MCP configuration has a conflicting deepkit-code entry"
            );
        }
    }

    #[test]
    fn refuses_malformed_shapes_and_duplicate_keys_at_every_depth() {
        for original in [
            "",
            "null",
            "[]",
            "false",
            "42",
            r#""private-placeholder""#,
            r#"{"mcpServers":null}"#,
            r#"{"mcpServers":[]}"#,
            r#"{"mcpServers":{},"mcpServers":{}}"#,
            r#"{"mcpServers":{"other":{},"other":{}}}"#,
            r#"{"mcpServers":{"other":{"env":{"X":1,"X":2}}}}"#,
            r#"{"metadata":[{"X":1,"\u0058":2}]}"#,
            r#"{"private-placeholder": }"#,
            r#"{} {}"#,
            "{/* comment */}",
        ] {
            let error = merge_mcp(Some(original.as_bytes())).unwrap_err();
            let rendered = format!("{error:#} {error:?}");
            assert!(!rendered.contains("private-placeholder"));
            assert!(error.to_string().starts_with("MCP configuration"));
        }
    }
}
