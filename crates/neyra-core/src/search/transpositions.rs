//! Exact, bounded reuse of a proven quiet-policy rejection. This is not a score
//! table: no heuristic miss, probability cutoff or incomplete graph is cached.
//! The coordinator transports immutable batches between private worker memories.
//! Full context bytes establish identity; a hash match alone never rejects work.

use super::{Plan, HEIGHT};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

const POLICY: &str = "neyra-exact-quiet-tt-v1";
const CAPACITY: usize = 2048;
const MAX_BATCH: usize = 32;
const MAX_WIRE_BYTES: usize = 16_384;
const FIXED_BYTES: usize = 1 + HEIGHT * 4 + 2 + 5;

pub(super) fn key(plan: &Plan, goal: u8) -> Vec<u8> {
    let mut key = Vec::with_capacity(FIXED_BYTES + plan.targets.len() * 9);
    key.push(goal);
    for row in plan.start.rows { key.extend(row.to_le_bytes()); }
    key.push(plan.start_spins);
    key.extend(plan.start_pieces.to_le_bytes());
    for row in plan.expected.rows { key.extend(row.to_le_bytes()); }
    key.extend([plan.expected_spins, u8::from(plan.terminal),
        u8::from(plan.assume_reflection), plan.targets.len() as u8]);
    // Target cells use the bag's original logical rows: compression history is
    // part of the assignment, not inferred from final occupancy or reflection.
    let mut targets: Vec<_> = plan.targets.iter().collect();
    targets.sort_unstable_by_key(|target| target.piece as u8);
    for target in targets {
        key.push(target.piece as u8);
        let mut cells = target.cells;
        cells.sort_unstable();
        for (x, y) in cells { key.extend([x, y]); }
    }
    key
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Batch {
    schema_version: u8,
    policy: String,
    entries: Vec<String>,
}

#[derive(Default)]
pub(super) struct QuietTable {
    keys: HashSet<Vec<u8>>,
    age: VecDeque<Vec<u8>>,
    outgoing: VecDeque<Vec<u8>>,
    pub hits: u64,
    pub stores: u64,
    pub imports: u64,
}

impl QuietTable {
    pub fn contains(&mut self, key: &[u8]) -> bool {
        let found = self.keys.contains(key);
        self.hits += u64::from(found);
        found
    }

    fn insert(&mut self, key: Vec<u8>) -> bool {
        if self.keys.contains(&key) { return false; }
        if self.keys.len() == CAPACITY {
            if let Some(old) = self.age.pop_front() { self.keys.remove(&old); }
        }
        self.age.push_back(key.clone());
        self.keys.insert(key);
        true
    }

    pub fn reject(&mut self, key: Vec<u8>) {
        if !self.insert(key.clone()) { return; }
        self.stores += 1;
        // Dropping an unsent entry only loses reuse. Workers never wait for a
        // peer result and can always perform their own exact verification.
        if self.outgoing.len() == MAX_BATCH { self.outgoing.pop_front(); }
        self.outgoing.push_back(key);
    }

    pub fn drain(&mut self, limit: usize) -> String {
        let count = limit.min(MAX_BATCH).min(self.outgoing.len());
        let entries = self.outgoing.drain(..count).map(|key| {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let mut out = String::with_capacity(key.len() * 2);
            for byte in key {
                out.push(HEX[(byte >> 4) as usize] as char);
                out.push(HEX[(byte & 15) as usize] as char);
            }
            out
        }).collect();
        serde_json::to_string(&Batch { schema_version: 1, policy: POLICY.into(), entries })
            .expect("bounded transposition batch")
    }

    pub fn observe(&mut self, json: &str) -> Result<(), String> {
        if json.len() > MAX_WIRE_BYTES { return Err("Transposition batch is too large".into()); }
        let batch: Batch = serde_json::from_str(json)
            .map_err(|_| "Invalid transposition batch".to_string())?;
        if batch.schema_version != 1 || batch.policy != POLICY || batch.entries.len() > MAX_BATCH {
            return Err("Unsupported transposition batch".into());
        }
        // Validate the entire batch before mutation. This internal same-build
        // channel is never a public certificate or a persisted remote input.
        let keys: Result<Vec<_>, _> = batch.entries.iter().map(|entry| decode(entry)).collect();
        for key in keys? {
            if self.insert(key) { self.imports += 1; }
        }
        Ok(())
    }
}

fn decode(value: &str) -> Result<Vec<u8>, String> {
    if value.len() < FIXED_BYTES * 2 || value.len() > (FIXED_BYTES + 7 * 9) * 2
        || value.len() % 2 != 0 {
        return Err("Invalid exact transposition key length".into());
    }
    let digit = |byte: u8| match byte {
        b'0'..=b'9' => Some(byte - b'0'), b'a'..=b'f' => Some(byte - b'a' + 10), _ => None,
    };
    let mut key = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let hi = digit(pair[0]).ok_or("Invalid exact transposition key")?;
        let lo = digit(pair[1]).ok_or("Invalid exact transposition key")?;
        key.push(hi * 16 + lo);
    }
    let targets = usize::from(key[FIXED_BYTES - 1]);
    if targets == 0 || targets > 7 || key.len() != FIXED_BYTES + targets * 9 {
        return Err("Invalid exact transposition target count".into());
    }
    Ok(key)
}
