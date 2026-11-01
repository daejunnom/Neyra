//! Small current-board CNN. Pure CPU inference, shared by native and WASM.
//! Outputs describe a censored training search, never physical certificates.

use crate::physics::{Board, HEIGHT, WIDTH};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Model {
    schema_version: u8,
    model_id: String,
    trained: bool,
    horizon_bags: u8,
    work_budget: u32,
    weights: HashMap<String, Vec<f32>>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BoardEvaluation {
    pub model_id: String,
    pub shape_preference: f64,
    pub observed_spins: f64,
    pub observed_choices: f64,
    pub horizon_bags: u8,
    pub work_budget: u32,
    pub adjustment: f64,
    pub experimental: bool,
    pub scope: &'static str,
}

fn model() -> Option<&'static Model> {
    static MODEL: OnceLock<Option<Model>> = OnceLock::new();
    MODEL.get_or_init(|| {
        let model: Model = serde_json::from_str(include_str!("../models/naga-cnn-v1.json")).ok()?;
        if model.schema_version != 1 || !model.trained || !(2..=3).contains(&model.horizon_bags)
            || model.work_budget == 0 { return None; }
        for (name, size) in [("conv1.weight", 108), ("conv1.bias", 6),
                             ("conv2.weight", 432), ("conv2.bias", 8),
                             ("hidden.weight", 1024), ("hidden.bias", 16),
                             ("output.weight", 48), ("output.bias", 3)] {
            let values = model.weights.get(name)?;
            if values.len() != size || values.iter().any(|x| !x.is_finite()) { return None; }
        }
        Some(model)
    }).as_ref()
}

fn encode(board: &Board) -> [f32; 2 * HEIGHT * WIDTH] {
    let reflected = board.rows.map(|row| row.reverse_bits() >> 6);
    let rows = if reflected < board.rows { reflected } else { board.rows };
    let mut input = [0.0; 2 * HEIGHT * WIDTH];
    for x in 0..WIDTH {
        let mut covered = false;
        for y in (0..HEIGHT).rev() {
            let index = y * WIDTH + x;
            if rows[y] & (1 << x) != 0 {
                input[index] = 1.0;
                covered = true;
            } else if covered {
                input[HEIGHT * WIDTH + index] = 1.0;
            }
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

/// Current occupancy/covered-air only. First-bag empty starts are outside the
/// training domain. The reflection canonicalization is a shape approximation.
pub fn raw_prediction(board: &Board) -> Option<[f32; 3]> {
    if board.rows.iter().all(|&r| r == 0) || board.rows.iter().any(|&r| r >= 1023) { return None; }
    let model = model()?;
    let w = &model.weights;
    let first = convolution(&encode(board), 2, 6, &w["conv1.weight"], &w["conv1.bias"]);
    let second = convolution(&first, 6, 8, &w["conv2.weight"], &w["conv2.bias"]);
    let mut pooled = [0.0; 64];
    for channel in 0..8 {
        for py in 0..4 {
            for px in 0..2 {
                let mut sum = 0.0;
                for y in py * 6..(py + 1) * 6 {
                    for x in px * 5..(px + 1) * 5 {
                        sum += second[channel * HEIGHT * WIDTH + y * WIDTH + x];
                    }
                }
                pooled[channel * 8 + py * 2 + px] = sum / 30.0;
            }
        }
    }
    let mut hidden = [0.0; 16];
    for (out, value) in hidden.iter_mut().enumerate() {
        let mut sum = w["hidden.bias"][out];
        for (i, input) in pooled.iter().enumerate() { sum += input * w["hidden.weight"][out * 64 + i]; }
        *value = sum.max(0.0);
    }
    let mut output = [0.0; 3];
    for (out, value) in output.iter_mut().enumerate() {
        let mut sum = w["output.bias"][out];
        for (i, input) in hidden.iter().enumerate() { sum += input * w["output.weight"][out * 16 + i]; }
        *value = sum;
    }
    output.iter().all(|x| x.is_finite()).then_some(output)
}

pub fn evaluate(board: &Board) -> Option<BoardEvaluation> {
    let raw = raw_prediction(board)?;
    let model = model()?;
    let shape = f64::from(raw[0]).clamp(-1.0, 1.0);
    let spins = (f64::from(raw[1]) * 6.0).clamp(0.0, f64::from(model.horizon_bags) * 6.0);
    let log_choices = (f64::from(raw[2]) * 4.0).clamp(0.0, 12.0);
    let adjustment = 12.0 * (0.4 * shape
        + 0.4 * (2.0 * (spins / 6.0).clamp(0.0, 1.0) - 1.0)
        + 0.2 * (2.0 * (log_choices / 4.0).clamp(0.0, 1.0) - 1.0));
    Some(BoardEvaluation {
        model_id: model.model_id.clone(), shape_preference: shape,
        observed_spins: spins, observed_choices: log_choices.exp_m1(),
        horizon_bags: model.horizon_bags, work_budget: model.work_budget,
        adjustment: adjustment.clamp(-12.0, 12.0), experimental: true,
        scope: "Prediction of bounded-policy observations; not a spin witness, exhaustive count, or construction probability.",
    })
}

/// A bounded ordering preference applied once at certified boundaries only.
pub fn boundary_adjustment(board: &Board) -> f64 {
    evaluate(board).map_or(0.0, |value| value.adjustment)
}
