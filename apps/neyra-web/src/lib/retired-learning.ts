const CLEANUP_MARKER = 'neyra-retired-learning-cleanup-v1';
const RETIRED_OUTBOX = 'neyra-contribution-outbox-v1';

/** Remove retired collection state by key/database name only. No payload is
 * opened, deserialized, or sent anywhere. Ordinary search preferences remain. */
export function discardRetiredLearningData(): void {
  let storage: Storage | undefined;
  let keysRemoved = false;
  try {
    storage = localStorage;
    if (storage.getItem(CLEANUP_MARKER) === 'done') return;
    const keys = Array.from({ length: storage.length }, (_, index) => storage!.key(index));
    for (const key of keys) {
      if (key?.startsWith('neyra-contribution-consent-') || key?.startsWith('neyra-focused-completions-v1:')) {
        storage.removeItem(key);
      }
    }
    keysRemoved = true;
  } catch { /* Storage restrictions must not block local search. */ }

  if (typeof indexedDB === 'undefined') return;
  try {
    const deletion = indexedDB.deleteDatabase(RETIRED_OUTBOX);
    deletion.onsuccess = () => {
      if (!keysRemoved) return;
      try { storage?.setItem(CLEANUP_MARKER, 'done'); } catch { /* Retry next visit. */ }
    };
    // An older open tab can postpone deletion. Do not mark that attempt as
    // complete; the request can finish after it closes, or retry next visit.
    deletion.onblocked = deletion.onerror = () => {};
  } catch { /* Retry next visit if IndexedDB is unavailable in this context. */ }
}
