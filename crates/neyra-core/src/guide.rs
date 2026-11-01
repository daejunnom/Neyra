//! Small, optional fixed-plan evaluation network.
//!
//! Predictions only order already legal completed bag plans. They are not
//! bounds, supply certificates, legality decisions, or completion evidence.

use crate::physics::{Piece, FULL_ROW, HEIGHT};
use crate::search::TargetOperation;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const INPUTS: usize = 547;
pub const HIDDEN: usize = 256;
pub const OUTPUTS: usize = 3;
pub const PARAMETER_COUNT: usize = INPUTS * HIDDEN + HIDDEN
    + 2 * (HIDDEN * HIDDEN + HIDDEN) + HIDDEN * OUTPUTS + OUTPUTS;
pub const SCHEMA: &str = "neyra-guide-model-v1";
pub const FEATURE_SCHEMA: &str = "neyra-guide-fixed-bag547-v1";
pub const RULE_PROFILE: &str = "neyra-sonic-srsplus-24-clear-hold-first-reflection-v1";
pub const ARCHITECTURE: &str = "dense547x256-residual256x2-head3-f32-v1";
const MAX_MODEL_BYTES: usize = 16 * 1024 * 1024;
const MAX_WEIGHT: f32 = 32.0;

/// A completed *fixed* plan's context, not an arbitrary queue observation.
/// Target cells are logical bag-local cells, before row compression.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GuideInput {
    pub start_rows: [u16; HEIGHT],
    pub end_rows: [u16; HEIGHT],
    pub targets: Vec<TargetOperation>,
    pub start_spins: u8,
    pub end_spins: u8,
    pub start_pieces: u16,
    pub terminal: bool,
}

impl GuideInput {
    pub fn validate(&self) -> Result<(), String> {
        if self.start_rows.iter().chain(&self.end_rows).any(|r| *r >= FULL_ROW)
            || self.start_spins >= 40 || self.end_spins > 40
            || self.end_spins < self.start_spins || self.end_spins - self.start_spins > 6
            || self.start_pieces > 125 || self.start_pieces % 7 != 0
            || self.targets.is_empty() || self.targets.len() > 7
            || self.terminal != (self.end_spins == 40)
            || (!self.terminal && self.targets.len() != 7) {
            return Err("Invalid fixed-bag guide board, counts, or terminal context".into());
        }
        let mut pieces = 0_u8;
        let mut occupied = [0_u16; 32];
        for target in &self.targets {
            if pieces & target.piece.bit() != 0 {
                return Err("Guide input repeats a piece label".into());
            }
            pieces |= target.piece.bit();
            for &(x, y) in &target.cells {
                if x >= 10 || y >= 32 || occupied[y as usize] & (1 << x) != 0 {
                    return Err("Guide input has invalid or overlapping logical cells".into());
                }
                occupied[y as usize] |= 1 << x;
            }
        }
        Ok(())
    }

    /// Canonical piece order I,J,L,O,S,T,Z; each four-cell set is sorted by
    /// (x,y). The Python trainer uses the same f32 normalization and layout.
    pub fn features(&self) -> Result<[f32; INPUTS], String> {
        self.validate()?;
        let mut values = [0.0_f32; INPUTS];
        for (offset, rows) in [(0, &self.start_rows), (240, &self.end_rows)] {
            for (y, row) in rows.iter().enumerate() {
                for x in 0..10 { values[offset + y * 10 + x] = f32::from(row & (1 << x) != 0); }
            }
        }
        for piece in Piece::ALL {
            if let Some(target) = self.targets.iter().find(|t| t.piece == piece) {
                let mut cells = target.cells;
                cells.sort_unstable();
                let p = piece as usize;
                for (i, (x, y)) in cells.into_iter().enumerate() {
                    values[480 + p * 8 + i * 2] = f32::from(x) / 9.0;
                    values[480 + p * 8 + i * 2 + 1] = f32::from(y) / 31.0;
                }
                values[536 + p] = 1.0;
            }
        }
        values[543] = f32::from(self.start_spins) / 40.0;
        values[544] = f32::from(self.end_spins) / 40.0;
        values[545] = f32::from(self.start_pieces) / 126.0;
        values[546] = f32::from(self.terminal);
        Ok(values)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GuideWeights {
    pub input_weight: Vec<f32>,
    pub input_bias: Vec<f32>,
    pub residual_in_weight: Vec<f32>,
    pub residual_in_bias: Vec<f32>,
    pub residual_out_weight: Vec<f32>,
    pub residual_out_bias: Vec<f32>,
    pub head_weight: Vec<f32>,
    pub head_bias: Vec<f32>,
}

impl GuideWeights {
    fn arrays(&self) -> [&[f32]; 8] {
        [&self.input_weight, &self.input_bias, &self.residual_in_weight,
            &self.residual_in_bias, &self.residual_out_weight, &self.residual_out_bias,
            &self.head_weight, &self.head_bias]
    }
    fn validate(&self) -> Result<(), String> {
        let sizes = [INPUTS * HIDDEN, HIDDEN, HIDDEN * HIDDEN, HIDDEN,
            HIDDEN * HIDDEN, HIDDEN, HIDDEN * OUTPUTS, OUTPUTS];
        if self.arrays().iter().zip(sizes).any(|(a, n)| a.len() != n)
            || self.arrays().iter().any(|a| a.iter().any(|v| !v.is_finite() || v.abs() > MAX_WEIGHT)) {
            return Err("Guide weights have an invalid shape or a nonfinite/out-of-range value".into());
        }
        Ok(())
    }
    fn digest(&self) -> String {
        let mut hash = Sha256::new();
        for values in self.arrays() {
            for value in values { hash.update(value.to_le_bytes()); }
        }
        format!("{:x}", hash.finalize())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GuideTraining {
    pub steps: u64,
    pub unique_plans: usize,
    pub train_groups: usize,
    pub validation_groups: usize,
    pub dataset_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GuideModel {
    pub schema_version: String,
    pub feature_schema: String,
    pub rules_profile: String,
    pub architecture: String,
    pub parameter_count: usize,
    pub model_id: String,
    pub weights_sha256: String,
    pub trained: bool,
    pub learning_complete: bool,
    pub proof_model: bool,
    pub training: GuideTraining,
    pub weights: GuideWeights,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GuidePrediction {
    pub model_id: String,
    pub quiet_safe_fraction: f32,
    pub construction_fraction: f32,
    pub bag_spins_fraction: f32,
    pub priority: f64,
    pub trained: bool,
    pub proof: bool,
}

fn dense(input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32]) {
    for (row, value) in output.iter_mut().enumerate() {
        let mut sum = bias[row];
        let w = &weights[row * input.len()..(row + 1) * input.len()];
        for (&x, &a) in input.iter().zip(w) { sum += x * a; }
        *value = sum;
    }
}
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) }
    else { let e = x.exp(); e / (1.0 + e) }
}

impl GuideModel {
    pub fn from_json(json: &str) -> Result<Self, String> {
        if json.len() > MAX_MODEL_BYTES { return Err("Guide model exceeds 16 MiB".into()); }
        let model: Self = serde_json::from_str(json).map_err(|e| format!("Invalid guide model: {e}"))?;
        model.validate()?;
        Ok(model)
    }
    pub fn validate(&self) -> Result<(), String> {
        self.weights.validate()?;
        if self.schema_version != SCHEMA || self.feature_schema != FEATURE_SCHEMA
            || self.rules_profile != RULE_PROFILE || self.architecture != ARCHITECTURE
            || self.parameter_count != PARAMETER_COUNT || self.learning_complete || self.proof_model
            || self.weights_sha256 != self.weights.digest()
            || self.model_id != format!("guide-{}", &self.weights_sha256[..20])
            || self.trained != (self.training.steps > 0)
            || self.training.train_groups.checked_add(self.training.validation_groups) != Some(self.training.unique_plans)
            || (self.trained && (self.training.train_groups == 0 || self.training.validation_groups == 0
                || self.training.dataset_sha256.is_none()))
            || self.training.dataset_sha256.as_ref().is_some_and(|s|
                s.len() != 64 || !s.bytes().all(|c| c.is_ascii_hexdigit())) {
            return Err("Guide schema, checksum, readiness, or training metadata is invalid".into());
        }
        Ok(())
    }
    /// Deterministic untrained initialization. It is intentionally not used by
    /// priority(), so merely loading an initializer cannot steer live search.
    pub fn initializer(seed: u32) -> Self {
        let mut rng = seed.max(1);
        let mut weights = |inputs: usize, outputs: usize, factor: f32| {
            let scale = (6.0_f32 / (inputs + outputs) as f32).sqrt() * factor;
            (0..inputs * outputs).map(|_| {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                ((rng as f64 / 4_294_967_296.0 * 2.0 - 1.0) as f32) * scale
            }).collect::<Vec<_>>()
        };
        let parameters = GuideWeights {
            input_weight: weights(INPUTS, HIDDEN, 1.0), input_bias: vec![0.0; HIDDEN],
            residual_in_weight: weights(HIDDEN, HIDDEN, 1.0), residual_in_bias: vec![0.0; HIDDEN],
            residual_out_weight: weights(HIDDEN, HIDDEN, 0.1), residual_out_bias: vec![0.0; HIDDEN],
            head_weight: weights(HIDDEN, OUTPUTS, 0.1), head_bias: vec![0.0; OUTPUTS],
        };
        let digest = parameters.digest();
        Self { schema_version: SCHEMA.into(), feature_schema: FEATURE_SCHEMA.into(),
            rules_profile: RULE_PROFILE.into(), architecture: ARCHITECTURE.into(),
            parameter_count: PARAMETER_COUNT, model_id: format!("guide-{}", &digest[..20]),
            weights_sha256: digest, trained: false, learning_complete: false, proof_model: false,
            training: GuideTraining { steps: 0, unique_plans: 0, train_groups: 0,
                validation_groups: 0, dataset_sha256: None }, weights: parameters }
    }
    pub fn model_id(&self) -> &str { &self.model_id }
    pub fn is_trained(&self) -> bool { self.trained }
    pub fn evaluate(&self, input: &GuideInput) -> Result<GuidePrediction, String> {
        let features = input.features()?;
        let mut h = [0.0; HIDDEN];
        dense(&features, &self.weights.input_weight, &self.weights.input_bias, &mut h);
        h.iter_mut().for_each(|v| *v = v.max(0.0));
        let mut r = [0.0; HIDDEN];
        dense(&h, &self.weights.residual_in_weight, &self.weights.residual_in_bias, &mut r);
        r.iter_mut().for_each(|v| *v = v.max(0.0));
        let mut residual = [0.0; HIDDEN];
        dense(&r, &self.weights.residual_out_weight, &self.weights.residual_out_bias, &mut residual);
        for (v, delta) in h.iter_mut().zip(residual) { *v = (*v + delta).max(0.0); }
        let mut logits = [0.0; OUTPUTS];
        dense(&h, &self.weights.head_weight, &self.weights.head_bias, &mut logits);
        if logits.iter().any(|v| !v.is_finite()) { return Err("Guide inference produced nonfinite logits".into()); }
        let [quiet, construction, spins] = logits.map(sigmoid);
        Ok(GuidePrediction { model_id: self.model_id.clone(), quiet_safe_fraction: quiet,
            construction_fraction: construction, bag_spins_fraction: spins,
            priority: 2.0 * f64::from(quiet) + f64::from(construction) + 0.25 * f64::from(spins),
            trained: self.trained, proof: false })
    }
    /// A failure removes only the optional ordering hint; search can continue
    /// with its existing score and exact physics/verifier.
    pub fn priority(&self, input: &GuideInput) -> Option<f64> {
        if !self.trained { return None; }
        self.evaluate(input).ok().map(|prediction| prediction.priority)
    }
}
