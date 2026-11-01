//! Portable GA runner. Keep generated checkpoints, models and measurement
//! reports outside the source repository.

use neyra_core::ga::{FrozenRunner, GaConfig, GaModel, Trainer};
use neyra_core::search::SearchSession;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

fn read(path: &str) -> Result<String, String> { fs::read_to_string(path).map_err(|e| format!("{path}: {e}")) }
fn write(path: &str, value: &str) -> Result<(), String> { fs::write(path, value).map_err(|e| format!("{path}: {e}")) }
fn number<T: std::str::FromStr>(args: &HashMap<String, String>, key: &str, default: T) -> Result<T, String> {
    args.get(key).map_or(Ok(default), |v| v.parse().map_err(|_| format!("Invalid --{key}")))
}
fn required<'a>(args: &'a HashMap<String, String>, key: &str) -> Result<&'a str, String> {
    args.get(key).map(String::as_str).ok_or_else(|| format!("Missing --{key}"))
}
fn config(args: &HashMap<String, String>) -> Result<GaConfig, String> {
    args.get("config").map_or(Ok(GaConfig::physical_defaults()), |p| GaConfig::from_json(&read(p)?))
}
fn emit(args: &HashMap<String, String>, value: &str) -> Result<(), String> {
    if let Some(path) = args.get("output") { write(path, value) } else { println!("{value}"); Ok(()) }
}

fn run() -> Result<(), String> {
    let mut raw = std::env::args().skip(1);
    let command = raw.next().unwrap_or_else(|| "help".into());
    if matches!(command.as_str(), "help" | "--help" | "-h") {
        println!("neyra-ga initializer [--config FILE] --output MODEL\nneyra-ga train [--config FILE] [--model MODEL | --resume CHECKPOINT] [--seconds N] [--work N] --output MODEL --checkpoint FILE [--report FILE]\nneyra-ga fork --resume CHECKPOINT [--config FILE] --checkpoint NEW_CHECKPOINT [--output SNAPSHOT]\nneyra-ga evaluate --model MODEL [--config FILE] [--seeds 1,2,3 | --seed-start N --episodes N] [--seconds N] [--work N] --output REPORT\nneyra-ga baseline [--config FILE] [--seed N] [--beam-width N] [--search-profile N] [--seconds N] [--work N] --output REPORT\nneyra-ga checkpoint --resume FILE [--work N] --checkpoint FILE [--output SNAPSHOT]\nNew runs default to v4 physical-only candidate generation; exact resumed legacy runs retain their original policy. Physical completions have unmeasured random-supply probability. Work units differ between policies; elapsed time and actual query counters are reported. Generated files belong outside the repository.");
        return Ok(());
    }
    let mut args = HashMap::new();
    while let Some(key) = raw.next() {
        let Some(key) = key.strip_prefix("--") else { return Err(format!("Unexpected argument {key}")); };
        let value = raw.next().ok_or_else(|| format!("Missing value for --{key}"))?;
        if args.insert(key.to_string(), value).is_some() { return Err(format!("Duplicate --{key}")); }
    }
    let allowed = ["config", "output", "resume", "checkpoint", "report", "seconds", "work", "model", "seeds", "seed-start", "episodes", "seed", "beam-width", "search-profile"];
    if let Some(key) = args.keys().find(|k| !allowed.contains(&k.as_str())) { return Err(format!("Unknown --{key}")); }
    let seconds: f64 = number(&args, "seconds", if command == "baseline" { 10.0 } else { 0.0 })?;
    let max_work: u64 = number(&args, "work", 0)?;
    if !seconds.is_finite() || seconds < 0.0 { return Err("--seconds must be finite and nonnegative".into()); }
    let start = Instant::now();
    let within = |work: u64| (seconds == 0.0 || start.elapsed().as_secs_f64() < seconds)
        && (max_work == 0 || work < max_work);
    match command.as_str() {
        "initializer" => emit(&args, &serde_json::to_string(&GaModel::initializer(config(&args)?)?).map_err(|e| e.to_string())?),
        "train" => {
            if args.contains_key("resume") && args.contains_key("config") { return Err("A resumed run uses its exact saved configuration".into()); }
            if args.contains_key("resume") && args.contains_key("model") { return Err("Choose exact resume or a new inherited-weight experiment".into()); }
            let mut trainer = if let Some(path) = args.get("resume") { Trainer::from_checkpoint_json(&read(path)?)? }
                else if let Some(path) = args.get("model") { Trainer::with_model(config(&args)?, &read(path)?)? }
                else { Trainer::new(config(&args)?)? };
            let initial_work = trainer.counters().work_units;
            let mut generation = 0;
            while !trainer.finished() && within(trainer.counters().work_units - initial_work) {
                trainer.advance(1)?;
                let current = trainer.completed_generations();
                if current != generation {
                    generation = current;
                    eprintln!("Completed generation {generation}; episodes {}; active {:.3}s", trainer.counters().evaluated_episodes, trainer.counters().elapsed_ms / 1000.0);
                }
            }
            write(required(&args, "checkpoint")?, &trainer.checkpoint_json()?)?;
            emit(&args, &trainer.champion_json()?)?;
            if let Some(path) = args.get("report") {
                let snapshot: serde_json::Value = serde_json::from_str(&trainer.snapshot_json()?).map_err(|e| e.to_string())?;
                write(path, &json!({"schemaVersion":"neyra-ga-training-report-v1","sourcePolicy":trainer.config().source_policy(),
                    "wallMs":start.elapsed().as_secs_f64()*1000.0,"runWorkUnits":trainer.counters().work_units-initial_work,
                    "stopReason":if trainer.finished(){"generations-complete"}else{"budget-exhausted"},"snapshot":snapshot}).to_string())?;
            }
            Ok(())
        }
        "fork" => {
            let trainer = Trainer::fork_checkpoint(config(&args)?, &read(required(&args,"resume")?)?)?;
            write(required(&args,"checkpoint")?, &trainer.checkpoint_json()?)?;
            emit(&args,&trainer.snapshot_json()?)
        }
        "evaluate" => {
            let model = GaModel::from_json(&read(required(&args, "model")?)?)?;
            let config = if args.contains_key("config") { config(&args)? } else { model.training_config.clone() };
            let seeds = if let Some(list) = args.get("seeds") {
                list.split(',').map(|s| s.parse::<u32>().map_err(|_| "Invalid --seeds list".to_string())).collect::<Result<Vec<_>, _>>()?
            } else {
                let first: u32 = number(&args, "seed-start", 1_000_003)?;
                let count: usize = number(&args, "episodes", 16)?;
                if count == 0 || count > 65_536 { return Err("--episodes must be in 1..65536".into()); }
                (0..count).map(|i| first.wrapping_add(i as u32)).collect()
            };
            let mut runner = FrozenRunner::new(model, config, seeds)?;
            while !runner.finished() && within(runner.counters().work_units) { runner.advance(1)?; }
            let mut report: serde_json::Value = serde_json::from_str(&runner.report_json()).map_err(|e| e.to_string())?;
            report["wallMs"] = (start.elapsed().as_secs_f64() * 1000.0).into();
            report["stopReason"] = if runner.finished(){"seed-roster-complete"}else{"budget-exhausted"}.into();
            emit(&args, &report.to_string())
        }
        "baseline" => {
            let mut cfg = config(&args)?.search_config();
            cfg.seed = number(&args, "seed", cfg.seed)?;
            cfg.beam_width = number(&args, "beam-width", cfg.beam_width)?;
            cfg.search_profile = number(&args, "search-profile", cfg.search_profile)?;
            let mut session = SearchSession::new(cfg.clone())?;
            let mut work = 0_u64;
            while within(work) {
                session.step(1); work += 1;
                if work % 64 == 0 && session.snapshot_compact().finished { break; }
            }
            let export: serde_json::Value = serde_json::from_str(&session.export_json()).map_err(|e| e.to_string())?;
            let snapshot = session.snapshot_compact();
            emit(&args, &json!({"schemaVersion":"neyra-ga-baseline-v1","sourcePolicy":"existing-structural-search",
                "config":cfg,"stepWork":work,"wallMs":start.elapsed().as_secs_f64()*1000.0,
                "snapshot":snapshot,"search":export,"deploymentReady":false,
                "workDefinition":"Existing SearchSession.step(1); not CPU-equivalent to a GA work unit. Compare elapsed-time budgets and report individual query counters."}).to_string())
        }
        "checkpoint" => {
            let mut trainer = Trainer::from_checkpoint_json(&read(required(&args, "resume")?)?)?;
            let initial = trainer.counters().work_units;
            while !trainer.finished() && max_work > 0 && trainer.counters().work_units - initial < max_work { trainer.advance(1)?; }
            write(required(&args, "checkpoint")?, &trainer.checkpoint_json()?)?;
            emit(&args, &trainer.snapshot_json()?)
        }
        _ => Err(format!("Unknown command {command}")),
    }
}

fn main() { if let Err(error) = run() { eprintln!("{error}"); std::process::exit(1); } }
