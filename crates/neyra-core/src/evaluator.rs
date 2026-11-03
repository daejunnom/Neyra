//! Independent Naga Eyes spin evaluator. All layers run on CPU in Rust/WASM.
//! Only current occupancy is input; outputs describe a censored search policy.

use crate::physics::{Board, HEIGHT, WIDTH};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::OnceLock;

pub const MODEL_TASK: &str = "naga-spin-observations-v2";
pub const NEXT_BAG_TASK: &str = "naga-next-bag-spin-v3";
pub const REFINED_BAG_TASK: &str = "naga-next-bag-spin-v4";
pub const PRECISION_BAG_TASK: &str = "naga-next-bag-spin-v5";
pub const MODEL_PARAMETERS: usize = 33_442;
const MODEL_JSON: &str = include_str!("../../../archives/pre-redesign-20261103/models/naga-spin-cnn-v2.json");
const MODEL_MAX_BYTES: usize = 2 * 1024 * 1024;
pub const WEIGHTS_HASH_ALGORITHM: &str = "sha256-f32le-neyraspin-v2";
pub(crate) const WEIGHT_LAYOUT: [(&str, usize); 10] = [
    ("conv1.weight", 216), ("conv1.bias", 24),
    ("conv2.weight", 6912), ("conv2.bias", 32),
    ("conv3.weight", 13824), ("conv3.bias", 48),
    ("hidden.weight", 12288), ("hidden.bias", 32),
    ("output.weight", 64), ("output.bias", 2),
];

pub(crate) fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(crate) fn validate_weights(
    weights: &HashMap<String, Vec<f32>>, layout: &[(&str, usize)], expected_hash: &str,
) -> Result<(), String> {
    if weights.len() != layout.len() || !valid_digest(expected_hash) {
        return Err("Unexpected model tensors or invalid weights digest".into());
    }
    let mut digest = Sha256::new();
    for &(name, size) in layout {
        let values = weights.get(name).ok_or_else(|| format!("Missing tensor {name}"))?;
        if values.len() != size || values.iter().any(|x| !x.is_finite() || x.abs() > 1000.0) {
            return Err(format!("Invalid tensor {name}"));
        }
        for value in values { digest.update(value.to_le_bytes()); }
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != expected_hash { return Err("Model weights hash mismatch".into()); }
    Ok(())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Architecture {
    name: String,
    input: Vec<usize>,
    channels: Vec<usize>,
    kernel: usize,
    padding: usize,
    pool_output: Vec<usize>,
    pool_kernel: Vec<usize>,
    hidden: usize,
    heads: usize,
    parameters: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Initialization {
    kind: String,
    source_model_id: String,
    source_weights_sha256: String,
    reused: Vec<String>,
    reset: Vec<String>,
    target_horizon_bags: u8,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ObservationContract {
    observation_version: Option<u8>,
    horizon_bags: u8,
    work_budget: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct TargetDerivation {
    schema_version: u8,
    policy: String,
    source_observation_version: u8,
    source_policy: String,
    target_horizon_bags: u8,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct TrainingObjective {
    schema_version: u8,
    kind: String,
    underestimate_weight: u8,
    overestimate_weight: u8,
    count_scale: u8,
}

/// Validated owner-produced weights. No search ranking or heuristic score is used.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpinModel {
    schema_version: u8,
    task: String,
    model_id: String,
    trained: bool,
    device: String,
    architecture: Architecture,
    horizon_bags: u8,
    target_horizon_bags: Option<u8>,
    work_budget: u32,
    label_config: Option<ObservationContract>,
    initialization: Option<Initialization>,
    target_derivation: Option<TargetDerivation>,
    training_objective: Option<TrainingObjective>,
    weights_sha256: String,
    weights_sha256_algorithm: String,
    output_names: Vec<String>,
    count_scale: u8,
    dataset_sha256: Option<String>,
    dataset_hash_algorithm: Option<String>,
    training_dataset_lineage: Option<Vec<String>>,
    search_ordering_enabled: bool,
    weights: HashMap<String, Vec<f32>>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BoardEvaluation {
    pub schema_version: u8,
    pub task: &'static str,
    pub model_id: String,
    /// Likelihood of observing at least one spin under this labeling policy.
    /// This is not fixed-plan construction success or an impossibility test.
    pub found_spin_likelihood: f64,
    pub observed_spins: f64,
    pub horizon_bags: u8,
    /// V3/V4 predict one inventory while actual follow-up measurement spans
    /// further bags. Absent on the unchanged historical v2 output contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_horizon_bags: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trained: Option<bool>,
    pub work_budget: u32,
    pub experimental: bool,
    pub scope: &'static str,
}

impl SpinModel {
    pub fn from_json(json: &str) -> Result<Self, String> {
        if json.len() > MODEL_MAX_BYTES { return Err("Spin model is too large".into()); }
        let model: Self = serde_json::from_str(json)
            .map_err(|e| format!("Invalid spin model JSON: {e}"))?;
        let a = &model.architecture;
        let next_bag = model.schema_version == 3 && model.task == NEXT_BAG_TASK
            && model.target_horizon_bags == Some(1) && model.target_derivation.is_none()
            && model.label_config.as_ref().is_some_and(|config| {
                config.observation_version == Some(3) && config.horizon_bags == model.horizon_bags
                    && config.work_budget == model.work_budget
            });
        let refined_bag = model.schema_version == 4 && model.task == REFINED_BAG_TASK
            && model.target_horizon_bags == Some(1) && model.horizon_bags == 2
            && model.label_config.as_ref().is_some_and(|config| {
                config.observation_version == Some(4) && config.horizon_bags == model.horizon_bags
                    && config.work_budget == model.work_budget
            })
            && model.target_derivation.as_ref().is_some_and(|derivation| {
                derivation.schema_version == 1 && derivation.policy == "best-replayed-next-bag-v1"
                    && derivation.source_observation_version == 4
                    && derivation.source_policy == "concentrated-next-bag-spin-v4"
                    && derivation.target_horizon_bags == 1
            });
        let legacy = model.schema_version == 2 && model.task == MODEL_TASK
            && model.target_horizon_bags.is_none() && model.target_derivation.is_none() && model.trained
            && model.label_config.as_ref().is_none_or(|config| {
                config.observation_version.is_none_or(|version| version == 2)
                    && config.horizon_bags == model.horizon_bags
                    && config.work_budget == model.work_budget
            });
        let precision_bag = model.schema_version == 5 && model.task == PRECISION_BAG_TASK
            && model.target_horizon_bags == Some(1) && model.horizon_bags == 1
            && model.label_config.as_ref().is_some_and(|config| {
                config.observation_version == Some(5) && config.horizon_bags == 1
                    && config.work_budget == model.work_budget
            })
            && model.target_derivation.as_ref().is_some_and(|derivation| {
                derivation.schema_version == 1 && derivation.policy == "best-replayed-next-bag-v1"
                    && derivation.source_observation_version == 5
                    && derivation.source_policy == "precision-single-bag-spin-v5"
                    && derivation.target_horizon_bags == 1
            })
            && model.training_objective.as_ref().is_some_and(|objective| {
                objective.schema_version == 1 && objective.kind == "asymmetric-smooth-l1-v1"
                    && objective.underestimate_weight == 2 && objective.overestimate_weight == 1
                    && objective.count_scale == 6
            });
        let honest_initialization = (next_bag || refined_bag || precision_bag) && model.initialization.as_ref().is_some_and(|init| {
            init.kind == (if precision_bag { "v2-feature-transfer-v5-head-reset" } else if refined_bag { "v2-feature-transfer-v4-head-reset" } else { "v2-feature-transfer-v3-head-reset" })
                && !init.source_model_id.is_empty() && init.source_model_id.len() <= 128
                && init.source_weights_sha256.len() == 64
                && init.source_weights_sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                && init.reused == ["conv1", "conv2", "conv3", "hidden"]
                && init.reset == ["output"] && init.target_horizon_bags == 1
        });
        if !(legacy || next_bag || refined_bag || precision_bag) || (!model.trained && !honest_initialization)
            || model.device != "cpu" || model.model_id.is_empty() || model.model_id.len() > 128
            || !(if precision_bag { model.horizon_bags == 1 } else { (2..=3).contains(&model.horizon_bags) })
            || !(1..=4_194_304).contains(&model.work_budget)
            || a.name != "neyra-spin-cnn-v2" || a.input != [1, HEIGHT, WIDTH]
            || a.channels != [24, 32, 48] || a.kernel != 3 || a.padding != 1
            || a.pool_output != [4, 2] || a.pool_kernel != [6, 5]
            || a.hidden != 32 || a.heads != 2 || a.parameters != MODEL_PARAMETERS
        {
            return Err("Unsupported spin-only CPU model contract".into());
        }
        if model.weights_sha256_algorithm != WEIGHTS_HASH_ALGORITHM || model.search_ordering_enabled
            || !matches!(model.dataset_hash_algorithm.as_deref().unwrap_or("sha256-raw-jsonl-v1"),
                "sha256-raw-jsonl-v1" | "sha256-neyra-canonical-projected-records-v1" | "sha256-neyra-refined-records-v1")
            || model.output_names != ["foundSpinLogit", "observedSpinsScaled"] || model.count_scale != 6
            || model.trained && !model.dataset_sha256.as_deref().is_some_and(valid_digest) {
            return Err("Invalid spin model output or provenance contract".into());
        }
        if let Some(lineage) = &model.training_dataset_lineage {
            let unique: std::collections::HashSet<_> = lineage.iter().collect();
            if lineage.is_empty() || lineage.len() > 256 || unique.len() != lineage.len()
                || lineage.iter().any(|hash| !valid_digest(hash))
                || !model.dataset_sha256.as_ref().is_some_and(|hash| lineage.contains(hash)) {
                return Err("Invalid spin model training lineage".into());
            }
        }
        validate_weights(&model.weights, &WEIGHT_LAYOUT, &model.weights_sha256)?;
        if !model.trained && ["output.weight", "output.bias"].iter()
            .any(|name| model.weights[*name].iter().any(|value| *value != 0.0)) {
            return Err("Untrained next-bag initialization must have a reset output head".into());
        }
        Ok(model)
    }

    pub fn raw_prediction(&self, board: &Board) -> Option<[f32; 2]> {
        let w = &self.weights;
        let pooled = occupancy_features(board, w)?;
        let mut hidden = [0.0; 32];
        let hidden_bias = &w["hidden.bias"];
        let hidden_weight = &w["hidden.weight"];
        for (out, value) in hidden.iter_mut().enumerate() {
            let mut sum = hidden_bias[out];
            for (i, input) in pooled.iter().enumerate() {
                sum += input * hidden_weight[out * 384 + i];
            }
            *value = sum.max(0.0);
        }
        let mut output = [0.0; 2];
        let output_bias = &w["output.bias"];
        let output_weight = &w["output.weight"];
        for (out, value) in output.iter_mut().enumerate() {
            let mut sum = output_bias[out];
            for (i, input) in hidden.iter().enumerate() {
                sum += input * output_weight[out * 32 + i];
            }
            *value = sum;
        }
        output.iter().all(|x| x.is_finite()).then_some(output)
    }

    pub fn model_id(&self) -> &str { &self.model_id }
    pub fn weights_sha256(&self) -> &str { &self.weights_sha256 }

    /// Experimental explicit use only. This does not assert learning completion,
    /// current private-corpus eligibility or better search; the owner pins a model.
    pub fn search_eligible(&self) -> bool {
        self.schema_version == 5 && self.trained && self.target_horizon_bags == Some(1)
    }

    pub fn boundary_quality(&self, board: &Board) -> Option<f64> {
        if !self.search_eligible() { return None; }
        let prediction = self.evaluate(board)?;
        Some(120.0 * (prediction.observed_spins + 0.05 * prediction.found_spin_likelihood))
    }

    pub fn evaluate(&self, board: &Board) -> Option<BoardEvaluation> {
        self.evaluate_with_raw(board).map(|(_, evaluation)| evaluation)
    }

    /// Produce raw and interpreted outputs with one forward pass. Native
    /// collection/validation clients often need both representations together.
    pub fn evaluate_with_raw(&self, board: &Board) -> Option<([f32; 2], BoardEvaluation)> {
        let raw = self.raw_prediction(board)?;
        let logit = f64::from(raw[0]).clamp(-80.0, 80.0);
        let next_bag = self.schema_version >= 3;
        Some((raw, BoardEvaluation {
            schema_version: self.schema_version,
            task: match self.schema_version { 5 => PRECISION_BAG_TASK, 4 => REFINED_BAG_TASK, 3 => NEXT_BAG_TASK, _ => MODEL_TASK }, model_id: self.model_id.clone(),
            found_spin_likelihood: 1.0 / (1.0 + (-logit).exp()),
            observed_spins: (f64::from(raw[1]) * 6.0).clamp(0.0,
                f64::from(self.target_horizon_bags.unwrap_or(self.horizon_bags)) * 6.0),
            horizon_bags: self.horizon_bags, work_budget: self.work_budget,
            target_horizon_bags: self.target_horizon_bags,
            trained: next_bag.then_some(self.trained),
            experimental: true,
            scope: "Current occupancy only. Predictions of censored follow-up spin observations; not construction probability, exhaustive feasibility, or a heuristic-score blend.",
        }))
    }
}

/// Shared occupancy encoder. The completion model adds progress only after this
/// tensor, preserving the first model's occupancy-only contract.
pub(crate) fn occupancy_features(board: &Board, w: &HashMap<String, Vec<f32>>) -> Option<[f32; 384]> {
    if board.rows.iter().all(|&r| r == 0) || board.rows.iter().any(|&r| r >= 1023) { return None; }
    let first = convolution(&encode(board), 1, 24, &w["conv1.weight"], &w["conv1.bias"]);
    let second = convolution(&first, 24, 32, &w["conv2.weight"], &w["conv2.bias"]);
    let third = convolution(&second, 32, 48, &w["conv3.weight"], &w["conv3.bias"]);
    let mut pooled = [0.0; 384];
    for channel in 0..48 {
        for py in 0..4 {
            for px in 0..2 {
                let mut sum = 0.0;
                for y in py * 6..(py + 1) * 6 {
                    for x in px * 5..(px + 1) * 5 {
                        sum += third[channel * HEIGHT * WIDTH + y * WIDTH + x];
                    }
                }
                pooled[channel * 8 + py * 2 + px] = sum / 30.0;
            }
        }
    }
    pooled.iter().all(|value| value.is_finite()).then_some(pooled)
}

fn model() -> Option<&'static SpinModel> {
    static MODEL: OnceLock<Option<SpinModel>> = OnceLock::new();
    MODEL.get_or_init(|| SpinModel::from_json(MODEL_JSON).ok()).as_ref()
}

/// Return the selected embedded model for owner-only training clients.
pub fn model_json() -> &'static str { MODEL_JSON }

fn encode(board: &Board) -> [f32; HEIGHT * WIDTH] {
    let reflected = board.rows.map(|row| row.reverse_bits() >> 6);
    let rows = if reflected < board.rows { reflected } else { board.rows };
    let mut input = [0.0; HEIGHT * WIDTH];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            input[y * WIDTH + x] = if rows[y] & (1 << x) != 0 { 1.0 } else { 0.0 };
        }
    }
    input
}

fn convolution(input: &[f32], cin: usize, cout: usize, weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; cout * HEIGHT * WIDTH];
    for out in 0..cout {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let mut sum = bias[out];
                for channel in 0..cin {
                    for ky in 0..3 {
                        let iy = y as isize + ky as isize - 1;
                        if !(0..HEIGHT as isize).contains(&iy) { continue; }
                        for kx in 0..3 {
                            let ix = x as isize + kx as isize - 1;
                            if !(0..WIDTH as isize).contains(&ix) { continue; }
                            let i = channel * HEIGHT * WIDTH + iy as usize * WIDTH + ix as usize;
                            let w = ((out * cin + channel) * 3 + ky) * 3 + kx;
                            sum += input[i] * weight[w];
                        }
                    }
                }
                output[out * HEIGHT * WIDTH + y * WIDTH + x] = sum.max(0.0);
            }
        }
    }
    output
}

pub fn raw_prediction(board: &Board) -> Option<[f32; 2]> {
    model()?.raw_prediction(board)
}

pub fn evaluate(board: &Board) -> Option<BoardEvaluation> {
    model()?.evaluate(board)
}

pub fn evaluate_with_raw(board: &Board) -> Option<([f32; 2], BoardEvaluation)> {
    model()?.evaluate_with_raw(board)
}
