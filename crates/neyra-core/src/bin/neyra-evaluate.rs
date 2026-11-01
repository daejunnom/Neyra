//! Stream current-board CNN predictions without running a continuation search.
use std::io::{self, BufRead, Write};

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model = match args.next().as_deref() {
        None => None,
        Some("--help" | "-h") => {
            println!("neyra-evaluate [--model v2-model.json]\nRead current-board JSONL from stdin and emit spin-only CPU predictions. No search or legacy score is used.");
            return Ok(());
        }
        Some("--model") => {
            let path = args.next().ok_or("--model needs a JSON path")?;
            if std::fs::metadata(&path).map_err(|e| e.to_string())?.len() > 2 * 1024 * 1024 {
                return Err("Model file is too large".into());
            }
            let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
            Some(neyra_core::evaluator::SpinModel::from_json(&json)?)
        }
        Some(_) => return Err("Unknown option; use --help".into()),
    };
    if args.next().is_some() { return Err("Unexpected extra arguments".into()); }
    let input = io::stdin();
    let mut output = io::BufWriter::new(io::stdout().lock());
    for (line_index, line) in input.lock().lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.len() > 8 * 1024 * 1024 { return Err("Input line too large".into()); }
        if line.trim().is_empty() { continue; }
        let value: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| format!("Line {}: {e}", line_index + 1))?;
        let input = value.get("input").unwrap_or(&value);
        let rows: [u16; neyra_core::physics::HEIGHT] = serde_json::from_value(input["rows"].clone())
            .map_err(|e| format!("Line {}: invalid board: {e}", line_index + 1))?;
        if rows.iter().any(|&r| r > neyra_core::physics::FULL_ROW) {
            return Err(format!("Line {}: row exceeds ten columns", line_index + 1));
        }
        let board = neyra_core::physics::Board { rows };
        let result = serde_json::json!({
            "id": input.get("id"),
            "rawOutput": model.as_ref().map_or_else(|| neyra_core::evaluator::raw_prediction(&board), |model| model.raw_prediction(&board)),
            "boardEvaluation": model.as_ref().map_or_else(|| neyra_core::evaluator::evaluate(&board), |model| model.evaluate(&board)),
        });
        writeln!(output, "{result}").map_err(|e| e.to_string())?;
    }
    output.flush().map_err(|e| e.to_string())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
