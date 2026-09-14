/**
 * Minimal typed event bus. Keeps engine subsystems decoupled: physics reports
 * contacts, gameplay listens; nothing imports the other directly.
 */

export type Listener<T> = (payload: T) => void;

/** The publishing half of an event bus; accept this instead of `EventBus`. */
export interface EventEmitter<Events extends Record<string, unknown>> {
  emit<K extends keyof Events>(type: K, payload: Events[K]): void;
}

export class EventBus<Events extends Record<string, unknown>> {
  private readonly map = new Map<keyof Events, Set<Listener<any>>>();

  on<K extends keyof Events>(type: K, listener: Listener<Events[K]>): () => void {
    let set = this.map.get(type);
    if (!set) {
      set = new Set();
      this.map.set(type, set);
    }
    set.add(listener as Listener<any>);
    return () => this.off(type, listener);
  }

  once<K extends keyof Events>(type: K, listener: Listener<Events[K]>): () => void {
    const off = this.on(type, (payload) => {
      off();
      listener(payload);
    });
    return off;
  }

  off<K extends keyof Events>(type: K, listener: Listener<Events[K]>): void {
    this.map.get(type)?.delete(listener as Listener<any>);
  }

  emit<K extends keyof Events>(type: K, payload: Events[K]): void {
    const set = this.map.get(type);
    if (!set || set.size === 0) return;
    // Copy: a listener may unsubscribe during dispatch.
    for (const listener of [...set]) listener(payload);
  }

  /**
   * Emit-only view of this bus.
   *
   * Subsystems should accept an `EventEmitter` rather than the concrete bus:
   * `EventBus<A & B>` is not assignable to `EventBus<A>` because `on`/`off`
   * consume `keyof Events` (making the class invariant), while emit-only use is
   * genuinely safe. This keeps physics decoupled from engine-level events.
   */
  asEmitter(): EventEmitter<Events> {
    return { emit: (type, payload) => this.emit(type, payload) };
  }

  clear(type?: keyof Events): void {
    if (type === undefined) this.map.clear();
    else this.map.delete(type);
  }

  get listenerCount(): number {
    let n = 0;
    for (const set of this.map.values()) n += set.size;
    return n;
  }
}
