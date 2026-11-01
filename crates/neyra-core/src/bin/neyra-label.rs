//! Native JSONL data collection. No GPU, network or browser runtime required.
use neyra_core::search::labels::{collect_board_label, BoardSample, LabelConfig};
use neyra_core::exploration::{collect_exploration_label, ExplorationConfig};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};

fn run() -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut config = LabelConfig::default();
    let mut exploration: Option<ExplorationConfig> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!("neyra-label [--input boards.jsonl] [--output labels.jsonl] [--config config.json] [--exploration exploration.json]\nOmitted input/output use stdin/stdout. Requires nonempty post-clear fields after at least two completed bags.\nConfig: horizonBags (2 or 3 measured bags), workBudget, beamWidth, profile, seed, maxNodes, attemptsPerNode, sourceRevision. Set observationVersion:3 for one-bag targets with deeper input-reachable observations; omit it to retain legacy v2 certified-parent collection.\nOptional private exploration: {{\"schemaVersion\":1,\"seed\":UINT32,\"ordinal\":UINT32}}. The explicit executed seed varies existing proposal ordering only; output has a separate exploration policy and cannot replace fixed-policy labels. The same exploration settings apply to each supplied board.\nSpin-only labels contain no Terrain teacher or opener prior. They are censored budgeted observations, not impossibility or completion certificates. V3 whole-bag input witnesses are not supply certificates.");
            return Ok(());
        }
        let value = args.next().ok_or_else(|| format!("Missing value for {arg}"))?;
        match arg.as_str() {
            "--input" => input = Some(value),
            "--output" => output = Some(value),
            "--config" => config = serde_json::from_reader(File::open(&value).map_err(|e| e.to_string())?)
                .map_err(|e| format!("Invalid config: {e}"))?,
            "--exploration" => exploration = Some(serde_json::from_reader(File::open(&value).map_err(|e| e.to_string())?)
                .map_err(|e| format!("Invalid exploration metadata: {e}"))?),
            _ => return Err(format!("Unknown option: {arg}")),
        }
    }
    if input.is_some() && input == output { return Err("Input and output must differ".into()); }
    let reader: Box<dyn BufRead> = match input {
        Some(path) => Box::new(BufReader::new(File::open(path).map_err(|e| e.to_string())?)),
        None => Box::new(BufReader::new(io::stdin())),
    };
    let writer: Box<dyn Write> = match output {
        Some(path) => Box::new(BufWriter::new(File::create(path).map_err(|e| e.to_string())?)),
        None => Box::new(BufWriter::new(io::stdout())),
    };
    let mut writer = writer;
    for (line, value) in reader.lines().enumerate() {
        let value = value.map_err(|e| e.to_string())?;
        if value.trim().is_empty() { continue; }
        let sample: BoardSample = serde_json::from_str(&value)
            .map_err(|e| format!("Line {}: invalid board sample: {e}", line + 1))?;
        if let Some(exploration) = &exploration {
            let label = collect_exploration_label(sample, config.clone(), exploration.clone())
                .map_err(|e| format!("Line {}: {e}", line + 1))?;
            serde_json::to_writer(&mut writer, &label).map_err(|e| e.to_string())?;
        } else {
            let label = collect_board_label(sample, config.clone())
                .map_err(|e| format!("Line {}: {e}", line + 1))?;
            serde_json::to_writer(&mut writer, &label).map_err(|e| e.to_string())?;
        }
        writer.write_all(b"\n").map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("neyra-label: {error}");
        std::process::exit(1);
    }
}
