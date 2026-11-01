//! Stream current-board CNN predictions without running a continuation search.
use std::io::{self, BufRead, Write};

fn run() -> Result<(), String> {
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
            "id": input.get("id"), "rawOutput": neyra_core::evaluator::raw_prediction(&board),
            "boardEvaluation": neyra_core::evaluator::evaluate(&board),
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
