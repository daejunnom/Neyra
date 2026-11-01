//! Private, reproducible proposal-order variation around the fixed collector.
//!
//! These observations never become fixed-policy training targets directly.
//! Their exact certified descendants can supply new boards for a subsequent
//! fixed-policy measurement. No model input, target, budget or rule is perturbed.

use crate::search::labels::{collect_board_label, BoardLabel, BoardSample, LabelConfig};
use serde::{Deserialize, Serialize};

pub const EXPLORATION_POLICY: &str = "neyra-seeded-bag-continuation-v2";
pub const NEXT_BAG_EXPLORATION_POLICY: &str = "seeded-next-bag-spin-v3";
pub const REFINED_BAG_EXPLORATION_POLICY: &str = "seeded-concentrated-next-bag-spin-v4";
pub const PRECISE_BAG_EXPLORATION_POLICY: &str = "seeded-precision-single-bag-spin-v5";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ExplorationConfig {
    pub schema_version: u8,
    /// The executed RNG seed, independent of the comparable base-policy seed.
    pub seed: u32,
    /// Caller-assigned attempt identity. It does not consume or enlarge a budget.
    pub ordinal: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExplorationMetadata {
    pub schema_version: u8,
    pub seed: u32,
    pub ordinal: u32,
    pub scope: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ExplorationLabel {
    #[serde(flatten)]
    pub label: BoardLabel,
    pub exploration: ExplorationMetadata,
}

pub fn collect_exploration_label(
    input: BoardSample,
    config: LabelConfig,
    exploration: ExplorationConfig,
) -> Result<ExplorationLabel, String> {
    if exploration.schema_version != 1 {
        return Err("Exploration metadata requires schemaVersion 1".into());
    }
    let mut executed = config.clone();
    executed.seed = u64::from(exploration.seed);
    let mut label = collect_board_label(input, executed)?;
    // Preserve the comparable base policy separately from the explicit seed
    // actually used by this private observation. The distinct policy name
    // prevents a seeded observation being mistaken for a fixed-policy label.
    label.config = config;
    label.policy = match label.schema_version {
        5 => PRECISE_BAG_EXPLORATION_POLICY,
        4 => REFINED_BAG_EXPLORATION_POLICY,
        3 => NEXT_BAG_EXPLORATION_POLICY,
        _ => EXPLORATION_POLICY,
    };
    Ok(ExplorationLabel {
        label,
        exploration: ExplorationMetadata {
            schema_version: 1,
            seed: exploration.seed,
            ordinal: exploration.ordinal,
            scope: "proposal-order-only",
        },
    })
}
