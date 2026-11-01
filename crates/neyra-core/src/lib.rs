pub mod physics;
pub mod evaluator;
pub mod exploration;
pub mod search;
mod srs_plus;
mod structure;
pub mod supply;

use search::{SearchConfig, SearchSession};
use wasm_bindgen::prelude::*;

/// Evaluate a current post-clear board without starting or advancing search.
#[wasm_bindgen]
pub fn evaluate_board(rows_json: &str) -> Result<String, JsValue> {
    let rows: [u16; physics::HEIGHT] = serde_json::from_str(rows_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid board: {e}")))?;
    if rows.iter().any(|&row| row > physics::FULL_ROW) {
        return Err(JsValue::from_str("Board cells exceed ten columns"));
    }
    serde_json::to_string(&evaluator::evaluate(&physics::Board { rows }))
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Evaluate with a downloaded v2 model without mixing any search/terrain score.
#[wasm_bindgen]
pub fn evaluate_board_model(rows_json: &str, model_json: &str) -> Result<String, JsValue> {
    let rows: [u16; physics::HEIGHT] = serde_json::from_str(rows_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid board: {e}")))?;
    if rows.iter().any(|&row| row > physics::FULL_ROW) {
        return Err(JsValue::from_str("Board cells exceed ten columns"));
    }
    let model = evaluator::SpinModel::from_json(model_json).map_err(|e| JsValue::from_str(&e))?;
    serde_json::to_string(&model.evaluate(&physics::Board { rows }))
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// One validated immutable downloaded model, reusable inside a learning worker.
/// JavaScript owns this handle and must call its generated `free()` method when
/// replacing the model. No global model selection or search state is changed.
#[wasm_bindgen]
pub struct SpinEvaluator {
    model: evaluator::SpinModel,
}

#[wasm_bindgen]
impl SpinEvaluator {
    #[wasm_bindgen(constructor)]
    pub fn new(model_json: &str) -> Result<SpinEvaluator, JsValue> {
        let model = evaluator::SpinModel::from_json(model_json)
            .map_err(|e| JsValue::from_str(&e))?;
        Ok(Self { model })
    }

    pub fn evaluate(&self, rows_json: &str) -> Result<String, JsValue> {
        let rows: [u16; physics::HEIGHT] = serde_json::from_str(rows_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid board: {e}")))?;
        if rows.iter().any(|&row| row > physics::FULL_ROW) {
            return Err(JsValue::from_str("Board cells exceed ten columns"));
        }
        serde_json::to_string(&self.model.evaluate(&physics::Board { rows }))
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// CPU training clients can bootstrap from these versioned embedded weights.
#[wasm_bindgen]
pub fn spin_model_json() -> String { evaluator::model_json().into() }

/// Run only the bounded deep-field labeling workflow, never a completion search.
/// Call this synchronous function from a dedicated browser worker.
#[wasm_bindgen]
pub fn label_board(sample_json: &str, config_json: &str) -> Result<String, JsValue> {
    if sample_json.len() > 65_536 || config_json.len() > 4096 {
        return Err(JsValue::from_str("Learning input is too large"));
    }
    let sample: search::labels::BoardSample = serde_json::from_str(sample_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid learning board: {e}")))?;
    let config: search::labels::LabelConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid learning budget: {e}")))?;
    let label = search::labels::collect_board_label(sample, config)
        .map_err(|e| JsValue::from_str(&e))?;
    serde_json::to_string(&label).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Privately vary existing proposal ordering with an explicit reproducible seed.
/// The returned policy is separate from fixed-policy training observations.
#[wasm_bindgen]
pub fn label_board_exploration(
    sample_json: &str,
    config_json: &str,
    exploration_json: &str,
) -> Result<String, JsValue> {
    if sample_json.len() > 65_536 || config_json.len() > 4096 || exploration_json.len() > 1024 {
        return Err(JsValue::from_str("Learning input is too large"));
    }
    let sample: search::labels::BoardSample = serde_json::from_str(sample_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid learning board: {e}")))?;
    let config: search::labels::LabelConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid learning budget: {e}")))?;
    let exploration: exploration::ExplorationConfig = serde_json::from_str(exploration_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid exploration metadata: {e}")))?;
    let label = exploration::collect_exploration_label(sample, config, exploration)
        .map_err(|e| JsValue::from_str(&e))?;
    serde_json::to_string(&label).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub struct NeyraEngine {
    session: SearchSession,
}

#[wasm_bindgen]
impl NeyraEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<NeyraEngine, JsValue> {
        let config: SearchConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid settings: {e}")))?;
        let session = SearchSession::new(config).map_err(|e| JsValue::from_str(&e))?;
        Ok(Self { session })
    }

    pub fn advance(&mut self, max_work: u32) {
        self.session.step(max_work.clamp(1, 64));
    }

    pub fn observe_benchmarks(&mut self, values_json: &str) -> Result<(), JsValue> {
        let values: Vec<u8> = serde_json::from_str(values_json)
            .map_err(|error| JsValue::from_str(&format!("Invalid benchmarks: {error}")))?;
        if values.len() > 101 { return Err(JsValue::from_str("Too many bag benchmarks")); }
        self.session.observe_benchmarks(&values);
        Ok(())
    }

    pub fn snapshot(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.session.snapshot())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn snapshot_compact(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.session.snapshot_compact())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn export_json(&self) -> String {
        self.session.export_json()
    }

    pub fn export_selected_json(&self, selection: &str) -> String {
        self.session.export_selected_json(selection)
    }
}
