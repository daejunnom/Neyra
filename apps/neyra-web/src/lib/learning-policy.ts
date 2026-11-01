import type { FocusedLabelConfig } from './types';

/** Refined observations keep a separate contract from the pinned public model. */
export const NEXT_BAG_LABEL_CONFIG: Readonly<FocusedLabelConfig> = Object.freeze({
  observationVersion: 4, horizonBags: 2, workBudget: 131072, beamWidth: 48,
  profile: 0, seed: 1, maxNodes: 96, attemptsPerNode: 16,
  sourceRevision: 'sha256:78bd36da629a393e140b433eee6eaa9f08ddcda80fd322a857d099a91b75acff'
});
