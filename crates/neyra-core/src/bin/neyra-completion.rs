//! Verify second-stage completed paths or evaluate a progress-conditioned CNN.
use std::io::{self, BufRead, Read, Write};
use neyra_core::{completion::{CompletionModel, CompletionPath, RULE_PROFILE, derive_target}, physics::{Board, HEIGHT}};

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model = match args.next().as_deref() {
        Some("--verify-paths") => None,
        Some("--model") => {
            let path = args.next().ok_or("--model requires a file")?;
            if std::fs::metadata(&path).map_err(|e| e.to_string())?.len() > 2 * 1024 * 1024 {
                return Err("Completion model is too large".into());
            }
            Some(CompletionModel::from_json(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)?)
        }
        Some("--help" | "-h") | None => {
            println!("neyra-completion --verify-paths | --model completion.json\nRead JSONL completed-path proofs, or inference inputs with rows, remainingSpins and rulesProfile. Unfinished paths cannot produce targets. Untrained models return null.");
            return Ok(());
        }
        _ => return Err("Unknown arguments; use --help".into()),
    };
    if args.next().is_some() { return Err("Unexpected extra arguments".into()); }
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout().lock());
    let mut reader = stdin.lock();
    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader.by_ref().take(8 * 1024 * 1024 + 1).read_line(&mut line).map_err(|e| e.to_string())?;
        if bytes == 0 { break; }
        if bytes > 8 * 1024 * 1024 { return Err("Input line exceeds bounded size".into()); }
        if line.trim().is_empty() { continue; }
        let value = if let Some(model) = &model {
            let input: serde_json::Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
            let rows: [u16; HEIGHT] = serde_json::from_value(input["rows"].clone()).map_err(|e| e.to_string())?;
            let remaining = input["remainingSpins"].as_u64().filter(|v| (1..=40).contains(v)).ok_or("Invalid remainingSpins")?;
            let rules = input["rulesProfile"].as_str().ok_or("Missing rulesProfile")?;
            if rules != RULE_PROFILE || rows.iter().any(|&r| r >= 1023) { return Err("Invalid rule profile or board".into()); }
            serde_json::to_value(model.evaluate(&Board { rows }, remaining as u8, rules)?).map_err(|e| e.to_string())?
        } else {
            let proof: CompletionPath = serde_json::from_str(&line).map_err(|e| format!("Invalid completion proof: {e}"))?;
            derive_target(&proof)?
        };
        writeln!(out, "{value}").map_err(|e| e.to_string())?;
    }
    out.flush().map_err(|e| e.to_string())
}

fn main() {
    if let Err(error) = run() { eprintln!("{error}"); std::process::exit(1); }
}
