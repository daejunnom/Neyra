//! Run the real, empty-board search with a bounded wall-clock budget.
//!
//! cargo run --release -p neyra-core --example search_smoke -- [seconds] [seed] [export.json]
//! The defaults are 60 seconds and SearchConfig's seed. All other parameters
//! are unchanged from the browser's default configuration.

use neyra_core::search::{ProgressSnapshot, SearchConfig, SearchSession};
use serde_json::json;
use std::io::{self, Write};
use std::time::{Duration, Instant};

fn report(snapshot: &ProgressSnapshot, elapsed: Duration, steps: u64, max_step: Duration) {
    let current = snapshot.current.as_ref();
    let best = snapshot.best.as_ref();
    println!(
        "{}",
        json!({
            "elapsedSeconds": (elapsed.as_secs_f64() * 100.0).round() / 100.0,
            "phase": snapshot.phase,
            "expanded": snapshot.expanded,
            "candidates": snapshot.candidates,
            "verified": snapshot.verified_candidates,
            "frontier": snapshot.frontier,
            "deepestSpins": snapshot.deepest_spins,
            "currentPlaced": current.map(|c| c.placed),
            "currentSpins": current.map(|c| c.spins),
            "currentProbability": current.map(|c| c.probability),
            "currentVerified": current.map(|c| c.verified),
            "currentComplete": current.map(|c| c.complete),
            "checked": snapshot.checked,
            "success": snapshot.success,
            "bestPlaced": best.map(|c| c.placed),
            "bestProbability": best.map(|c| c.probability),
            "workSteps": steps,
            "maxWorkStepMs": (max_step.as_secs_f64() * 100_000.0).round() / 100.0,
            "finished": snapshot.finished,
        })
    );
    let _ = io::stdout().flush();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let seconds = args
        .next()
        .map(|s| s.parse::<u64>())
        .transpose()?
        .unwrap_or(60);
    if !(1..=60).contains(&seconds) {
        return Err("seconds must be between 1 and 60".into());
    }
    let mut config = SearchConfig::default();
    if let Some(seed) = args.next() {
        config.seed = seed.parse()?;
    }
    let export_path = args.next();
    if args.next().is_some() {
        return Err("usage: search_smoke [seconds 1-60] [seed] [export.json]".into());
    }
    println!(
        "{}",
        json!({ "config": config, "wallClockBudgetSeconds": seconds })
    );
    let mut session = SearchSession::new(config)?;
    let started = Instant::now();
    let budget = Duration::from_secs(seconds);
    let mut next_report = Duration::from_secs(5);
    let mut steps = 0;
    let mut max_step = Duration::ZERO;
    report(&session.snapshot(), started.elapsed(), steps, max_step);
    loop {
        if started.elapsed() >= budget {
            break;
        }
        let before_step = Instant::now();
        session.step(1);
        max_step = max_step.max(before_step.elapsed());
        steps += 1;
        let elapsed = started.elapsed();
        if elapsed >= next_report {
            let snapshot = session.snapshot();
            report(&snapshot, elapsed, steps, max_step);
            next_report += Duration::from_secs(5);
            if snapshot.finished {
                break;
            }
        }
    }
    report(&session.snapshot(), started.elapsed(), steps, max_step);
    if let Some(path) = export_path {
        std::fs::write(path, session.export_json())?;
    }
    Ok(())
}
