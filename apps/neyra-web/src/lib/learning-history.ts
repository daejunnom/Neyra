import { NEXT_BAG_LABEL_CONFIG } from './learning-policy';

const LIMIT = 4096;
const STORAGE_KEY = `neyra-focused-completions-v1:${JSON.stringify(NEXT_BAG_LABEL_CONFIG)}`;
const keyFor = (rows: number[]) => rows.map(row => row.toString(16).padStart(3, '0')).join('');

/** Reuse only completed, acknowledged deterministic work with the same full
 * collector policy. Orientation is preserved; this is not a reflected proof. */
export class LearningHistory {
  private completed = new Set<string>();

  refresh() {
    try {
      const value: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]');
      if (Array.isArray(value)) for (const key of value.slice(-LIMIT)) {
        if (typeof key === 'string' && /^[0-9a-f]{72}$/.test(key)) this.completed.add(key);
      }
    } catch { /* Storage denial leaves this tab's history usable. */ }
    while (this.completed.size > LIMIT) this.completed.delete(this.completed.values().next().value!);
  }

  has(rows: number[]) { return this.completed.has(keyFor(rows)); }

  acknowledge(rows: number[]) {
    this.refresh();
    const key = keyFor(rows);
    this.completed.delete(key);
    this.completed.add(key);
    while (this.completed.size > LIMIT) this.completed.delete(this.completed.values().next().value!);
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify([...this.completed])); }
    catch { /* A network receipt still completes this tab's work. */ }
  }
}

export const learningHistory = new LearningHistory();
