//! Portable guide-model initialization and CPU inference; no network access.

use neyra_core::guide::{GuideInput, GuideModel};
use std::{collections::HashMap, fs};

fn read(path: &str) -> Result<String, String> {
    let size = fs::metadata(path).map_err(|e| e.to_string())?.len();
    if size > 16 * 1024 * 1024 { return Err("Input file exceeds 16 MiB".into()); }
    fs::read_to_string(path).map_err(|e| e.to_string())
}
fn run() -> Result<(), String> {
    let mut argv = std::env::args().skip(1);
    let command = argv.next().unwrap_or_else(|| "help".into());
    if matches!(command.as_str(), "help" | "--help" | "-h") {
        println!("neyra-guide init [--seed N] --output MODEL\nneyra-guide evaluate --model MODEL --input PLAN [--output PREDICTION]\nneyra-guide inspect --model MODEL\nPredictions order completed fixed-bag plans; they are never probability certificates. Use learning/guide_train.py for CPU/CUDA training.");
        return Ok(());
    }
    let mut args = HashMap::<String, String>::new();
    while let Some(key) = argv.next() {
        let key = key.strip_prefix("--").ok_or("Expected --option")?.to_string();
        let value = argv.next().ok_or("Missing option value")?;
        if !["model", "input", "output", "seed"].contains(&key.as_str()) || args.insert(key, value).is_some() {
            return Err("Unknown or duplicate option".into());
        }
    }
    let required = |name: &str| args.get(name).map(String::as_str).ok_or_else(|| format!("Missing --{name}"));
    let result = match command.as_str() {
        "init" => {
            let seed = args.get("seed").map_or(Ok(1), |s| s.parse::<u32>().map_err(|e| e.to_string()))?;
            serde_json::to_value(GuideModel::initializer(seed)).map_err(|e| e.to_string())?
        }
        "evaluate" => {
            let model = GuideModel::from_json(&read(required("model")?)?)?;
            let input: GuideInput = serde_json::from_str(&read(required("input")?)?).map_err(|e| e.to_string())?;
            serde_json::to_value(model.evaluate(&input)?).map_err(|e| e.to_string())?
        }
        "inspect" => {
            let model = GuideModel::from_json(&read(required("model")?)?)?;
            serde_json::json!({"modelId":model.model_id(), "trained":model.is_trained(),
                "parameterCount":model.parameter_count,"training":model.training,"proof":false})
        }
        _ => return Err("Unknown command".into()),
    };
    let json = serde_json::to_string(&result).map_err(|e| e.to_string())?;
    if let Some(path) = args.get("output") { fs::write(path, json).map_err(|e| e.to_string())?; }
    else { println!("{json}"); }
    Ok(())
}
fn main() {
    if let Err(error) = run() { eprintln!("{error}"); std::process::exit(1); }
}
