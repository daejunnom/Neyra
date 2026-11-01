//! Second-stage CNN for observed completion cost, separate from one-bag spin
//! prediction. Progress is an explicit input and the immutable rule profile
//! selects the meaning. A completed representative path is not an optimum or
//! an all-supplies certificate. Unfinished paths do not provide exact costs.

use crate::evaluator::{occupancy_features, valid_digest, validate_weights};
use crate::physics::{Board, Input, Piece};
use crate::search::labels::BoardSample;
use crate::supply::{clearing_only_finish, ClearOnlyLanguage, ClearingEdge, ClearingNode, ALL_PIECES};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub const TASK: &str = "naga-completion-cost-v1";
pub const POLICY: &str = "input-replayed-completion-cost-v1";
pub const RULE_PROFILE: &str = "neyra-srs-plus-24-spin-single-clear-hold-v1";
pub const HASH_ALGORITHM: &str = "sha256-f32le-neyracompletion-v1";
pub const PARAMETERS: usize = 33_441;
pub const COST_SCALE: f64 = 140.0;
const MAX_LOCKS: usize = 700;
const LAYOUT: [(&str, usize); 10] = [
    ("conv1.weight", 216), ("conv1.bias", 24),
    ("conv2.weight", 6912), ("conv2.bias", 32),
    ("conv3.weight", 13824), ("conv3.bias", 48),
    ("hidden.weight", 12320), ("hidden.bias", 32),
    ("output.weight", 32), ("output.bias", 1),
];

pub fn architecture() -> serde_json::Value {
    serde_json::json!({"name":"neyra-completion-cnn-v1", "input":[1,24,10],
        "channels":[24,32,48], "kernel":3, "padding":1,
        "poolOutput":[4,2], "poolKernel":[6,5], "context":["remainingSpins/40"],
        "hidden":32, "heads":1, "parameters":PARAMETERS})
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Initialization {
    kind: String,
    source_model_id: String,
    source_weights_sha256: String,
    source_task: String,
    reused: Vec<String>,
    reset: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletionModel {
    schema_version: u8,
    task: String,
    model_id: String,
    trained: bool,
    device: String,
    architecture: serde_json::Value,
    rules_profile: String,
    observation_policy: String,
    output_names: Vec<String>,
    cost_scale: f64,
    search_ordering_enabled: bool,
    initialization: Initialization,
    weights_sha256_algorithm: String,
    weights_sha256: String,
    dataset_sha256: Option<String>,
    weights: HashMap<String, Vec<f32>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletionEvaluation {
    pub schema_version: u8,
    pub task: &'static str,
    pub model_id: String,
    pub rules_profile: &'static str,
    pub remaining_spins: u8,
    pub observed_remaining_locks: f64,
    pub trained: bool,
    pub experimental: bool,
    pub scope: &'static str,
}

impl CompletionModel {
    pub fn from_json(json: &str) -> Result<Self, String> {
        if json.len() > 2 * 1024 * 1024 { return Err("Completion model is too large".into()); }
        let model: Self = serde_json::from_str(json).map_err(|e| format!("Invalid completion model: {e}"))?;
        let init = &model.initialization;
        if model.schema_version != 1 || model.task != TASK || model.device != "cpu"
            || model.model_id.is_empty() || model.model_id.len() > 128
            || model.architecture != architecture() || model.rules_profile != RULE_PROFILE
            || model.observation_policy != POLICY || model.cost_scale != COST_SCALE
            || model.output_names != ["observedRemainingLocksScaled"] || model.search_ordering_enabled
            || init.kind != "v5-occupancy-features-completion-head-reset-v1"
            || init.source_task != crate::evaluator::PRECISION_BAG_TASK
            || init.source_model_id.is_empty() || init.source_model_id.len() > 128
            || !valid_digest(&init.source_weights_sha256)
            || init.reused != ["conv1", "conv2", "conv3", "hidden.occupancy"]
            || init.reset != ["hidden.remainingSpins", "output"]
            || model.weights_sha256_algorithm != HASH_ALGORITHM
            || model.trained && !model.dataset_sha256.as_deref().is_some_and(valid_digest) {
            return Err("Unsupported progress-conditioned completion contract".into());
        }
        validate_weights(&model.weights, &LAYOUT, &model.weights_sha256)?;
        if !model.trained && (model.weights["output.weight"].iter().any(|v| *v != 0.0)
            || model.weights["output.bias"].iter().any(|v| *v != 0.0)
            || (0..32).any(|out| model.weights["hidden.weight"][out * 385 + 384] != 0.0)) {
            return Err("An untrained completion artifact must keep new context/output weights reset".into());
        }
        Ok(model)
    }

    pub fn evaluate(&self, board: &Board, remaining_spins: u8, rules_profile: &str)
        -> Result<Option<CompletionEvaluation>, String> {
        if !(1..=40).contains(&remaining_spins) || rules_profile != self.rules_profile {
            return Err("Completion inference requires remainingSpins in 1..40 and the exact rule profile".into());
        }
        if !self.trained { return Ok(None); }
        let Some(features) = occupancy_features(board, &self.weights) else { return Ok(None); };
        let context = f32::from(remaining_spins) / 40.0;
        let mut hidden = [0.0_f32; 32];
        for (out, value) in hidden.iter_mut().enumerate() {
            let mut sum = self.weights["hidden.bias"][out];
            for (i, input) in features.iter().enumerate() {
                sum += input * self.weights["hidden.weight"][out * 385 + i];
            }
            sum += context * self.weights["hidden.weight"][out * 385 + 384];
            *value = sum.max(0.0);
        }
        let mut raw = self.weights["output.bias"][0];
        for (i, value) in hidden.iter().enumerate() { raw += value * self.weights["output.weight"][i]; }
        if !raw.is_finite() { return Ok(None); }
        Ok(Some(CompletionEvaluation {
            schema_version: 1, task: TASK, model_id: self.model_id.clone(), rules_profile: RULE_PROFILE,
            remaining_spins, observed_remaining_locks: (f64::from(raw) * COST_SCALE).clamp(0.0, MAX_LOCKS as f64),
            trained: true, experimental: true,
            scope: "Observed cost of completed representative paths under the recorded collection policy; not optimal cost, construction probability, or a pruning bound",
        }))
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct CompletionMove {
    pub piece: Piece,
    pub inputs: Vec<Input>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct CompletionBag {
    pub queue: [u8; 7],
    pub moves: Vec<CompletionMove>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct CompletionPath {
    pub schema_version: u8,
    pub task: String,
    pub policy: String,
    pub rules_profile: String,
    pub source_revision: String,
    /// Identical collection settings must stay separate from other policies.
    pub collection_policy_sha256: String,
    pub input: BoardSample,
    pub completed_spins: u8,
    pub bags: Vec<CompletionBag>,
}

/// Re-execute every physical lock and the actual representative supply/hold
/// sequence. No fresh search, no acceptance based on a client `verified` flag.
pub fn derive_target(path: &CompletionPath) -> Result<serde_json::Value, String> {
    let input = &path.input;
    let source = &input.source;
    let split = source.extra.get("split").and_then(|v| v.as_str()).unwrap_or("");
    let group = source.extra.get("group").and_then(|v| v.as_str()).unwrap_or("");
    if path.schema_version != 1 || path.task != TASK || path.policy != POLICY || path.rules_profile != RULE_PROFILE
        || path.source_revision.is_empty() || path.source_revision.len() > 128
        || !valid_digest(&path.collection_policy_sha256) || path.completed_spins >= 40
        || !(2..=100).contains(&source.bag_boundary) || source.kind != "search"
        || source.family.is_empty() || source.family.len() > 256
        || source.reference.is_empty() || source.reference.len() > 4096
        || input.id.is_empty() || input.id.len() > 256 || group.is_empty() || group.len() > 256
        || !matches!(split, "train" | "validation" | "test")
        || input.rows.iter().all(|&r| r == 0) || input.rows.iter().any(|&r| r >= 1023)
        || path.bags.is_empty() || path.bags.len() > 100 {
        return Err("Invalid complete-path provenance, task, progress or board".into());
    }
    let expected_cells = u32::from(source.bag_boundary) * 28;
    if expected_cells < u32::from(path.completed_spins) * 10
        || expected_cells - u32::from(path.completed_spins) * 10
            != input.rows.iter().map(|row| row.count_ones()).sum::<u32>() {
        return Err("Completion progress disagrees with the actual bag-boundary material".into());
    }
    let mut board = Board { rows: input.rows };
    let mut spins = path.completed_spins;
    let mut locks = 0_usize;
    for (bag_index, bag) in path.bags.iter().enumerate() {
        let mut supply = 0_u8;
        for &piece in &bag.queue {
            if piece >= 7 || supply & (1 << piece) != 0 { return Err("Invalid complete bag supply".into()); }
            supply |= 1 << piece;
        }
        if bag.moves.is_empty() || bag.moves.len() > 7 { return Err("Invalid completion bag length".into()); }
        let mut nodes = vec![];
        let mut used = 0;
        let mut order = 0_u32;
        for (index, step) in bag.moves.iter().enumerate() {
            if spins == 40 || used & step.piece.bit() != 0 || step.inputs.len() > 512 {
                return Err("Repeated label or moves after the exact completion".into());
            }
            let placement = board.replay(step.piece, &step.inputs).ok_or("Completion input replay failed")?;
            let result = board.apply(&placement).ok_or("Completion placement failed")?;
            if result.lines > 1 || result.lines == 1 && (!result.spin || step.piece == Piece::O) {
                return Err("Completion contains a non-Spin-Single clear".into());
            }
            let spawn_mask = Piece::ALL.iter().fold(0, |mask, &piece| mask | if board.can_spawn(piece) { piece.bit() } else { 0 });
            nodes.push(ClearingNode { accepting: false, used, spawn_mask,
                edges: vec![ClearingEdge { piece: step.piece as u8, child: index + 1, clears: result.lines == 1 }] });
            used |= step.piece.bit();
            order |= u32::from(step.piece as u8) << (index * 3);
            board = result.board;
            spins += result.lines;
            locks += 1;
        }
        if spins != 40 && used != ALL_PIECES { return Err("An unfinished intermediate bag is not a boundary".into()); }
        if spins == 40 && bag_index + 1 != path.bags.len() { return Err("Extra bags after completion".into()); }
        nodes.push(ClearingNode { accepting: true, used, spawn_mask: 0, edges: vec![] });
        let language = ClearOnlyLanguage::new(nodes, used);
        if clearing_only_finish(&language, &bag.queue).finish != Some((bag.moves.len() as u8, order)) {
            return Err("Representative supply violates the clear-only hold policy".into());
        }
    }
    if spins != 40 || locks > MAX_LOCKS { return Err("Only actual complete 40-Spin-Single paths provide exact observed costs".into()); }
    let canonical = serde_json::to_vec(&serde_json::to_value(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let digest = format!("{:x}", Sha256::digest(&canonical));
    Ok(serde_json::json!({
        "schemaVersion":1, "kind":"completion-cost-target", "task":TASK, "policy":POLICY,
        "rulesProfile":RULE_PROFILE, "input":input, "completedSpins":path.completed_spins,
        "remainingSpins":40-path.completed_spins, "targets":{"observedRemainingLocks":locks},
        "collectionPolicySha256":path.collection_policy_sha256,
        "evidence":{"proofSha256":digest, "sourceRevision":path.source_revision,
            "physicalReplay":true, "representativeSupplyVerified":true, "allSuppliesVerified":false},
        "scope":"Observed completed path cost; no optimum, all-supplies or incomplete-path cost claim"
    }))
}
