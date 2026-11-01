//! Lossless content-addressed checkpoint storage. Full episode and supply
//! evidence is kept once even when several optimizer/archive slots refer to
//! it. This changes neither the runtime state nor the legacy v2 format.

use super::digest;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Evidence {
    schema_version: String,
    episodes: Map<String, Value>,
    proofs: Map<String, Value>,
}

fn is_proof(value: &Value) -> bool {
    ["canonicalCosts", "effectiveCosts", "orderCodes", "representativeFrames", "targets", "bagIndex"]
        .iter().all(|key| value.get(*key).is_some())
}
fn is_episode(value: &Value) -> bool {
    ["bagProofs", "placements", "rawSpins", "rawPieces", "qualifiedFrames", "seed"]
        .iter().all(|key| value.get(*key).is_some())
}
fn hash(value: &Value) -> Result<String, String> {
    serde_json::to_vec(value).map(|bytes| digest(&bytes)).map_err(|e| e.to_string())
}

fn intern(value: &mut Value, evidence: &mut Evidence) -> Result<(), String> {
    match value {
        Value::Array(values) => for item in values { intern(item, evidence)?; },
        Value::Object(values) => for item in values.values_mut() { intern(item, evidence)?; },
        _ => {}
    }
    let (tag, table) = if is_proof(value) { ("$gaProof", &mut evidence.proofs) }
        else if is_episode(value) { ("$gaEpisode", &mut evidence.episodes) }
        else { return Ok(()); };
    let key = hash(value)?;
    let object = std::mem::replace(value, serde_json::json!({ tag: key }));
    table.entry(key).or_insert(object);
    Ok(())
}

pub(super) fn compact(state: &mut Value) -> Result<(), String> {
    let mut evidence = Evidence { schema_version: "neyra-ga-evidence-v1".into(), ..Evidence::default() };
    intern(state, &mut evidence)?;
    state["gaEvidence"] = serde_json::to_value(evidence).map_err(|e| e.to_string())?;
    Ok(())
}

fn resolve(value: &mut Value, episodes: &Map<String, Value>, proofs: &Map<String, Value>,
    expanded_bytes: &mut usize) -> Result<(), String> {
    if let Value::Object(object) = value {
        let reference = object.get("$gaEpisode").map(|id| (id, episodes))
            .or_else(|| object.get("$gaProof").map(|id| (id, proofs)));
        if let Some((key, table)) = reference {
            if object.len() != 1 { return Err("Invalid GA evidence reference shape".into()); }
            let key = key.as_str().ok_or("Invalid GA evidence reference")?;
            let item = table.get(key).ok_or("Missing GA checkpoint evidence")?;
            *expanded_bytes = expanded_bytes.saturating_add(serde_json::to_vec(item).map_err(|e|e.to_string())?.len());
            if *expanded_bytes > 128 * 1024 * 1024 { return Err("Expanded GA checkpoint exceeds the bounded evidence limit".into()); }
            *value = item.clone();
            // Table entries are resolved and checked before the state. No
            // recursive references or untrusted graph cycles are followed.
            return Ok(());
        }
    }
    match value {
        Value::Array(values) => for item in values { resolve(item, episodes, proofs, expanded_bytes)?; },
        Value::Object(values) => for item in values.values_mut() { resolve(item, episodes, proofs, expanded_bytes)?; },
        _ => {}
    }
    Ok(())
}

fn has_reference(value: &Value) -> bool {
    match value {
        Value::Object(object) => object.contains_key("$gaEpisode") || object.contains_key("$gaProof")
            || object.values().any(has_reference),
        Value::Array(values) => values.iter().any(has_reference),
        _ => false,
    }
}

pub(super) fn expand(state: &mut Value) -> Result<(), String> {
    let encoded = state.as_object_mut().ok_or("Invalid GA state object")?.remove("gaEvidence");
    let Some(encoded) = encoded else { return Ok(()); };
    let mut evidence: Evidence = serde_json::from_value(encoded).map_err(|e|format!("Invalid GA evidence table: {e}"))?;
    if evidence.schema_version != "neyra-ga-evidence-v1" || evidence.episodes.len() > 1024 || evidence.proofs.len() > 16384 {
        return Err("Invalid GA evidence schema or capacity".into());
    }
    for (key, proof) in &evidence.proofs {
        if !is_proof(proof) || *key != hash(proof)? || has_reference(proof) {
            return Err("Invalid content-addressed GA proof".into());
        }
    }
    let empty = Map::new(); let mut expanded_bytes = 0;
    for (key, episode) in &mut evidence.episodes {
        if !is_episode(episode) || *key != hash(episode)? { return Err("Invalid content-addressed GA episode".into()); }
        resolve(episode, &empty, &evidence.proofs, &mut expanded_bytes)?;
        if has_reference(episode) { return Err("Nested GA episode references are forbidden".into()); }
    }
    resolve(state, &evidence.episodes, &evidence.proofs, &mut expanded_bytes)?;
    if has_reference(state) { return Err("Unresolved GA checkpoint evidence".into()); }
    Ok(())
}
