pub mod physics;
pub mod search;
mod srs_plus;
mod structure;
pub mod supply;

use search::{SearchConfig, SearchSession};
use wasm_bindgen::prelude::*;

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

    pub fn snapshot(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.session.snapshot())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn export_json(&self) -> String {
        self.session.export_json()
    }

    pub fn export_selected_json(&self, selection: &str) -> String {
        self.session.export_selected_json(selection)
    }
}
