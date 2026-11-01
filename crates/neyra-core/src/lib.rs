pub mod physics;
pub mod evaluator;
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
