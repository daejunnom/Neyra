/** Only observations captured under explicit consent are retained here. No
 * credentials are stored; withdrawal removes every pending payload. */
export interface StoredObservation { id: string; body: string; key: string; }

export class ObservationOutbox {
  private database: Promise<IDBDatabase | null> | null = null;
  private operations: Promise<unknown> = Promise.resolve();

  constructor(private destination: string) {}

  private open(): Promise<IDBDatabase | null> {
    this.database ??= new Promise(resolve => {
      if (typeof indexedDB === 'undefined') { resolve(null); return; }
      let completed = false;
      const finish = (database: IDBDatabase | null) => {
        if (completed) { database?.close(); return; }
        completed = true;
        clearTimeout(timeout);
        resolve(database);
      };
      const timeout = setTimeout(() => finish(null), 4000);
      let request: IDBOpenDBRequest;
      try { request = indexedDB.open('neyra-contribution-outbox-v1', 1); }
      catch { finish(null); return; }
      request.onupgradeneeded = () => request.result.createObjectStore('observations', { keyPath: 'storageKey' });
      request.onsuccess = () => {
        request.result.onversionchange = () => request.result.close();
        finish(request.result);
      };
      request.onerror = () => finish(null);
      request.onblocked = () => finish(null);
    });
    return this.database;
  }

  private run<T>(work: (store: IDBObjectStore, result: (value: T) => void) => void,
    fallback: T): Promise<T> {
    const operation = this.operations.then(async () => {
      const database = await this.open();
      if (!database) return fallback;
      return new Promise<T>(resolve => {
        let value = fallback;
        try {
          const transaction = database.transaction('observations', 'readwrite');
          transaction.oncomplete = () => resolve(value);
          transaction.onerror = transaction.onabort = () => resolve(fallback);
          work(transaction.objectStore('observations'), result => { value = result; });
        } catch { resolve(fallback); }
      });
    });
    this.operations = operation.catch(() => undefined);
    return operation.catch(() => fallback);
  }

  read(): Promise<StoredObservation[]> {
    return this.run<StoredObservation[]>((store, result) => {
      const request = store.getAll();
      request.onsuccess = () => result(request.result
        .filter(entry => entry.destination === this.destination)
        .sort((a, b) => a.createdAt - b.createdAt)
        .map(({ id, body, key }) => ({ id, body, key })));
    }, []);
  }

  put(entry: StoredObservation): Promise<boolean> {
    return this.run((store, result) => {
      store.put({ ...entry, destination: this.destination,
        storageKey: `${this.destination}:${entry.id}`, createdAt: Date.now() });
      result(true);
    }, false);
  }

  remove(id: string): Promise<boolean> {
    return this.run((store, result) => {
      store.delete(`${this.destination}:${id}`);
      result(true);
    }, false);
  }

  clear(): Promise<boolean> {
    return this.run((store, result) => {
      const request = store.openCursor();
      request.onsuccess = () => {
        const cursor = request.result;
        if (!cursor) { result(true); return; }
        if (cursor.value.destination === this.destination) cursor.delete();
        cursor.continue();
      };
    }, false);
  }
}
