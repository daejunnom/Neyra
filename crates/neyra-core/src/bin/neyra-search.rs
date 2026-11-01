//! Bounded native search portfolio. Optional fitted guides order proposals;
//! independent CPU lanes retain their own exact fixed-plan certificates.
use neyra_core::search::{SearchConfig, SearchSession};
use num_bigint::BigUint;
use serde_json::{json, Value};
use std::{collections::{BTreeMap, HashMap, HashSet}, fs, sync::Arc, thread, time::{Duration, Instant}};

fn read(path: &str) -> Result<String, String> {
    let metadata = fs::metadata(path).map_err(|e| format!("{path}: {e}"))?;
    if metadata.len() > 32 * 1024 * 1024 { return Err("Input exceeds 32 MiB".into()); }
    fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))
}
fn write(path: &str, value: &Value) -> Result<(), String> {
    let temporary = format!("{path}.pending-{}", std::process::id());
    fs::write(&temporary, value.to_string()).map_err(|e| e.to_string())?;
    fs::rename(&temporary, path).map_err(|e| e.to_string())
}
fn number<T: std::str::FromStr>(args: &HashMap<String, String>, key: &str, default: T) -> Result<T, String> {
    args.get(key).map_or(Ok(default), |v| v.parse().map_err(|_| format!("Invalid --{key}")))
}
fn fraction(point: &Value) -> Option<(BigUint, BigUint)> {
    Some((BigUint::parse_bytes(point["probabilityNumerator"].as_str()?.as_bytes(), 10)?,
        BigUint::parse_bytes(point["probabilityDenominator"].as_str()?.as_bytes(), 10)?))
}
fn merge_frontiers(lanes: &[Value], cap: u16) -> Value {
    let mut grid: BTreeMap<(u8, u16), Value> = BTreeMap::new();
    for lane in lanes {
        let Some(groups) = lane["snapshot"]["probabilityFrontier"].as_array() else { continue; };
        for group in groups {
            let Some(goal) = group["targetSpins"].as_u64() else { continue; };
            let Some(points) = group["points"].as_array() else { continue; };
            for point in points {
                let Some((n, d)) = fraction(point) else { continue; };
                let Some(cost) = point["pieces"].as_u64() else { continue; };
                for budget in cost as u16..=cap {
                    let key = (goal as u8, budget);
                    let replace = grid.get(&key).is_none_or(|old| fraction(old)
                        .is_some_and(|(old_n, old_d)| &n * old_d > old_n * &d));
                    if replace {
                        let mut selected = point.clone();
                        selected["pieces"] = budget.into();
                        selected["worker"] = lane["worker"].clone();
                        grid.insert(key, selected);
                    }
                }
            }
        }
    }
    let mut result = vec![];
    for goal in 1..=40 {
        let mut points = vec![];
        let mut previous: Option<(BigUint, BigUint)> = None;
        for (_, point) in grid.range((goal, 0)..=(goal, cap)) {
            let Some((n, d)) = fraction(point) else { continue; };
            if previous.as_ref().is_some_and(|(p, q)| &n * q <= p * &d) { continue; }
            points.push(point.clone());
            let certain = n == d;
            previous = Some((n, d));
            if certain { break; }
        }
        if !points.is_empty() { result.push(json!({"targetSpins":goal,"complete":goal==40,"points":points})); }
    }
    Value::Array(result)
}

fn run() -> Result<(), String> {
    let mut raw = std::env::args().skip(1);
    let mut args = HashMap::new();
    while let Some(key) = raw.next() {
        if matches!(key.as_str(), "--help" | "-h") {
            println!("neyra-search [--config FILE] [--ga-model FILE] [--guide-model FILE] [--seconds 10] [--threads 1] [--seed 1] --output REPORT [--dataset-output DATASET]\nThe probability frontier includes the first bag. No probability-budget selector is used. CPU physics and independent P7 proofs stay active with or without learned models. Generated reports/models/datasets belong outside the repository.");
            return Ok(());
        }
        let key = key.strip_prefix("--").ok_or("Expected --option")?.to_string();
        let value = raw.next().ok_or("Missing option value")?;
        if args.insert(key, value).is_some() { return Err("Duplicate option".into()); }
    }
    let allowed = ["config","ga-model","guide-model","seconds","threads","seed","output","dataset-output"];
    if let Some(key) = args.keys().find(|k| !allowed.contains(&k.as_str())) { return Err(format!("Unknown --{key}")); }
    let output = args.get("output").ok_or("Missing --output")?;
    let seconds: f64 = number(&args,"seconds",10.0)?;
    let threads: usize = number(&args,"threads",1)?;
    if !seconds.is_finite() || !(0.01..=86_400.0).contains(&seconds) || !(1..=32).contains(&threads) {
        return Err("Use seconds in 0.01..86400 and threads in 1..32".into());
    }
    let mut config: SearchConfig = if let Some(path) = args.get("config") {
        serde_json::from_str(&read(path)?).map_err(|e| e.to_string())?
    } else { SearchConfig::default() };
    config.probability_frontier = true;
    config.ignore_first_bag_probability = false;
    config.seed = number(&args,"seed",config.seed)?;
    // Optional guidance must not make exact CPU search unavailable. File I/O
    // failures are surfaced alongside contract failures reported by each lane.
    let mut model_load_errors = vec![];
    let mut load_optional = |key: &str| args.get(key).and_then(|path| match read(path) {
        Ok(encoded) => Some(encoded),
        Err(error) => {
            model_load_errors.push(json!({"model":key,"error":error,"fallback":"existing-cpu-search"}));
            None
        }
    });
    let ga = Arc::new(load_optional("ga-model"));
    let guide = Arc::new(load_optional("guide-model"));
    let collect = args.contains_key("dataset-output");
    let started = Instant::now();
    let deadline = started + Duration::from_secs_f64(seconds);
    let mut handles = vec![];
    for worker in 0..threads {
        let mut local = config.clone();
        local.search_lane = worker as u16;
        local.search_lanes = threads as u16;
        local.search_profile = (worker % 4) as u8;
        let ga = Arc::clone(&ga); let guide = Arc::clone(&guide);
        handles.push((worker, thread::spawn(move || -> Result<(Value, Vec<Value>), String> {
            let mut session = SearchSession::with_guides(local, ga.as_deref(), guide.as_deref())?;
            if collect { session.enable_guide_collection(4096); }
            let mut work = 0_u64;
            while Instant::now() < deadline {
                session.step(8); work += 8;
                if work % 256 == 0 && session.snapshot_compact().finished { break; }
            }
            let snapshot = session.snapshot_compact();
            let mut ids = HashSet::new();
            let mut routes = vec![];
            for group in &snapshot.probability_frontier {
                for point in &group.points {
                    let id = &point.candidate.id;
                    if ids.insert(id.clone()) {
                        let data: Value = serde_json::from_str(&session.export_selected_json(&format!("candidate:{id}")))
                            .map_err(|e| e.to_string())?;
                        routes.push(json!({"id":id,"worker":worker,"data":data}));
                    }
                }
            }
            Ok((json!({"worker":worker,"stepWork":work,"snapshot":snapshot,"routes":routes}), session.take_guide_samples()))
        })));
    }
    let mut lanes = vec![]; let mut errors = vec![]; let mut dataset = vec![];
    for (worker, handle) in handles {
        match handle.join() {
            Ok(Ok((lane, samples))) => { lanes.push(lane); dataset.extend(samples); }
            Ok(Err(error)) => errors.push(json!({"worker":worker,"error":error})),
            Err(_) => errors.push(json!({"worker":worker,"error":"worker-panic"})),
        }
    }
    if lanes.is_empty() { return Err(format!("All search workers failed: {}", Value::Array(errors))); }
    let frontier = merge_frontiers(&lanes, config.max_pieces);
    let report = json!({"schemaVersion":"neyra-search-frontier-v1","sourcePolicy":"legal-placement-multipath-independent-P7-v1",
        "requestedSeconds":seconds,"wallMs":started.elapsed().as_secs_f64()*1000.0,
        "threadsRequested":threads,"threadsCompleted":lanes.len(),"errors":errors,"modelLoadErrors":model_load_errors,
        "probabilityContext":"all-bags-included;clear-only-hold;accepted-first-bag-reflection-model",
        "frontierMeaning":"Best observed probability of fully constructing an independent fixed plan that achieves at least targetSpins within pieces. This is a lower bound on a lower spin target's first-hit probability; intermediate-lock first-hit CDFs are not measured. The 40-spin terminal cost CDF is measured. No union across plans or workers, no global optimality claim.",
        "probabilityFrontier":frontier,"lanes":lanes,"learningComplete":false});
    write(output, &report)?;
    if let Some(path) = args.get("dataset-output") {
        write(path, &json!({"schemaVersion":"neyra-guide-dataset-v1","rulesProfile":neyra_core::guide::RULE_PROFILE,
            "records":dataset,"sourcePolicy":"neyra-exact-search-frontier-v1"}))?;
    }
    eprintln!("Completed {} of {threads} CPU lanes; wrote {output}", report["threadsCompleted"]);
    Ok(())
}
fn main() { if let Err(error) = run() { eprintln!("{error}"); std::process::exit(1); } }
