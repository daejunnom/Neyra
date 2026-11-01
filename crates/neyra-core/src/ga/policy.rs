//! A small context-conditioned residual policy for probability-free GA runs.
//!
//! The first 24 coefficients are the unchanged legacy linear evaluator. New
//! coefficients see the proposed piece, the remaining bag inventory, phase,
//! and a local patch around the actual landing. Products make this nonlinear
//! in the observed state without asking a small population to evolve a large
//! dense neural network. These measurements order legal moves; they do not
//! assert future reachability, a successful supply, or a completed game.

use super::{features as base_features, FEATURE_COUNT, FEATURE_NAMES, FEATURE_SCHEMA, INITIAL_WEIGHTS};
use crate::physics::{Board, Piece, Placement, HEIGHT, WIDTH};
use crate::supply::ALL_PIECES;

pub(super) const CONTEXT_FEATURE_COUNT: usize = 256;
pub(super) const CONTEXT_FEATURE_SCHEMA: &str = "neyra-ga-context256-v1";

const PIECE_GEOMETRY: [usize; 12] = [0, 1, 4, 6, 7, 11, 12, 13, 14, 15, 16, 23];
const INVENTORY_GEOMETRY: [usize; 8] = [0, 4, 7, 11, 12, 13, 15, 16];
const PIECE_NAMES: [&str; 7] = ["I", "J", "L", "O", "S", "T", "Z"];

pub(super) fn initializer(expanded: bool) -> Vec<f64> {
    let mut weights = INITIAL_WEIGHTS.to_vec();
    if expanded { weights.resize(CONTEXT_FEATURE_COUNT, 0.0); }
    weights
}

pub(super) fn validate_weights(weights: &[f64]) -> Result<(), String> {
    if !matches!(weights.len(), FEATURE_COUNT | CONTEXT_FEATURE_COUNT)
        || weights.iter().any(|w| !w.is_finite() || w.abs() > 16.0) {
        return Err("GA weights must contain 24 or 256 finite coefficients in [-16, 16]".into());
    }
    Ok(())
}

/// Expansion preserves every original action score. Shrinking would discard
/// fitted context and therefore requires a separate explicit projection tool.
pub(super) fn upgrade_weights(weights: &[f64], expanded: bool) -> Result<Vec<f64>, String> {
    validate_weights(weights)?;
    if !expanded && weights.len() != FEATURE_COUNT {
        return Err("A contextual GA model cannot silently become a 24-coefficient model".into());
    }
    let mut result = weights.to_vec();
    if expanded { result.resize(CONTEXT_FEATURE_COUNT, 0.0); }
    Ok(result)
}

pub(super) fn feature_schema(count: usize) -> Option<&'static str> {
    match count {
        FEATURE_COUNT => Some(FEATURE_SCHEMA),
        CONTEXT_FEATURE_COUNT => Some(CONTEXT_FEATURE_SCHEMA),
        _ => None,
    }
}

pub(super) fn feature_names(count: usize) -> Option<Vec<String>> {
    feature_schema(count)?;
    let mut names: Vec<_> = FEATURE_NAMES.iter().map(|s| (*s).to_string()).collect();
    if count == FEATURE_COUNT { return Some(names); }
    for piece in PIECE_NAMES {
        for index in PIECE_GEOMETRY {
            names.push(format!("piece{piece}*{}", FEATURE_NAMES[index]));
        }
    }
    for piece in PIECE_NAMES {
        for index in INVENTORY_GEOMETRY {
            names.push(format!("remaining{piece}*{}", FEATURE_NAMES[index]));
        }
    }
    for phase in ["bagProgress", "remainingLockFraction", "spinProgress"] {
        for index in PIECE_GEOMETRY {
            names.push(format!("{phase}*{}", FEATURE_NAMES[index]));
        }
    }
    for dy in -2..6 {
        for dx in -1..5 {
            names.push(format!("beforeLandingPatch[{dx},{dy}]"));
        }
    }
    for extra in ["rotation0", "rotation1", "rotation2", "rotation3", "landingX",
        "landingBottom", "holes*maximumHeight", "bumpiness*maximumHeight"] {
        names.push(extra.into());
    }
    debug_assert_eq!(names.len(), CONTEXT_FEATURE_COUNT);
    Some(names)
}

/// Measure context at the decision being scored. `used_mask` is the bag's
/// already placed pieces, and `pieces`/`spins` are counts before this move.
/// New context is entirely observable at this point; no future labels enter.
pub(super) fn features(before: &Board, after: &Board, placement: &Placement, lines: u8,
    used_mask: u8, spins: u8, pieces: u16, max_pieces: u16) -> [f64; CONTEXT_FEATURE_COUNT] {
    let base = base_features(before, after, placement, lines);
    expanded_features(base, before, placement, used_mask, spins, pieces, max_pieces)
}

fn expanded_features(base: [f64; FEATURE_COUNT], before: &Board, placement: &Placement,
    used_mask: u8, spins: u8, pieces: u16, max_pieces: u16) -> [f64; CONTEXT_FEATURE_COUNT] {
    let mut out = [0.0; CONTEXT_FEATURE_COUNT];
    out[..FEATURE_COUNT].copy_from_slice(&base);
    let mut at = FEATURE_COUNT;
    for piece in Piece::ALL {
        let active = f64::from(placement.pose.piece == piece);
        for index in PIECE_GEOMETRY { out[at] = active * base[index]; at += 1; }
    }
    let remaining = ALL_PIECES & !used_mask & !placement.pose.piece.bit();
    for piece in Piece::ALL {
        let active = f64::from(remaining & piece.bit() != 0);
        for index in INVENTORY_GEOMETRY { out[at] = active * base[index]; at += 1; }
    }
    let phases = [
        f64::from((used_mask & ALL_PIECES).count_ones()) / 6.0,
        f64::from(max_pieces.saturating_sub(pieces)) / f64::from(max_pieces.max(1)),
        f64::from(spins) / 40.0,
    ];
    for phase in phases {
        for index in PIECE_GEOMETRY { out[at] = phase * base[index]; at += 1; }
    }
    let cells = placement.pose.cells();
    let left = cells.iter().map(|(x, _)| *x).min().unwrap();
    let bottom = cells.iter().map(|(_, y)| *y).min().unwrap();
    for dy in -2..6 {
        for dx in -1..5 {
            // `occupied` intentionally includes the actual floor and walls.
            // The patch is only geometry, never a body-clearance certificate.
            out[at] = f64::from(before.occupied(left + dx, bottom + dy)); at += 1;
        }
    }
    for rotation in 0..4 { out[at] = f64::from(placement.pose.rotation == rotation); at += 1; }
    out[at] = 2.0 * f64::from(left) / (WIDTH - 1) as f64 - 1.0; at += 1;
    out[at] = f64::from(bottom) / HEIGHT as f64; at += 1;
    out[at] = base[4] * base[1]; at += 1;
    out[at] = base[6] * base[1]; at += 1;
    debug_assert_eq!(at, CONTEXT_FEATURE_COUNT);
    out
}

/// Callers validate model shape and finite coefficients at the boundary.
/// Zero residual weights produce the exact legacy accumulation order.
pub(super) fn score(weights: &[f64], before: &Board, after: &Board, placement: &Placement,
    lines: u8, used_mask: u8, spins: u8, pieces: u16, max_pieces: u16) -> f64 {
    debug_assert!(matches!(weights.len(), FEATURE_COUNT | CONTEXT_FEATURE_COUNT));
    let base = base_features(before, after, placement, lines);
    let initial = base.iter().zip(weights).map(|(f, w)| f * w).sum::<f64>();
    if weights.len() == FEATURE_COUNT { return initial; }
    let context = expanded_features(base, before, placement, used_mask, spins, pieces, max_pieces);
    initial + context[FEATURE_COUNT..].iter().zip(&weights[FEATURE_COUNT..])
        .map(|(f, w)| f * w).sum::<f64>()
}
