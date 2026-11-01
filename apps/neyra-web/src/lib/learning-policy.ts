import type { FocusedLabelConfig } from './types';

/** Refined observations keep a separate contract from the pinned public model. */
export const NEXT_BAG_LABEL_CONFIG: Readonly<FocusedLabelConfig> = Object.freeze({
  observationVersion: 5, horizonBags: 1, workBudget: 524288, beamWidth: 48,
  profile: 0, seed: 1, maxNodes: 96, attemptsPerNode: 1,
  sourceRevision: 'sha256:f0a00092a8b50034ede285d4afe519497b3f372cf25f13a1eef701f3df11a0dc'
});
