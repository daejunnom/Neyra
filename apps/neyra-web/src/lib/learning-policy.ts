import type { FocusedLabelConfig } from './types';

/** New observations are versioned independently of the pinned two-bag model. */
export const NEXT_BAG_LABEL_CONFIG: Readonly<FocusedLabelConfig> = Object.freeze({
  observationVersion: 3, horizonBags: 2, workBudget: 131072, beamWidth: 48,
  profile: 0, seed: 1, maxNodes: 96, attemptsPerNode: 6,
  sourceRevision: 'sha256:fc94c1a31763439465ab676db88ed5ba746696885451a1848b39f8adf70788d4'
});
