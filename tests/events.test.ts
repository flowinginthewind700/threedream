/**
 * EventBus specs.
 *
 * The bus is the seam that keeps physics from importing gameplay, so its
 * guarantees are load-bearing rather than incidental. Three are worth pinning
 * because each one, if broken, produces a failure far from the cause:
 *
 * - unsubscribing during dispatch must not skip or double-call other listeners
 *   (the emit loop copies the set for exactly this reason);
 * - `once` must fire exactly one time, including when the payload repeats;
 * - `asEmitter()` must be emit-only, so a subsystem cannot subscribe to or
 *   clear a bus it was handed.
 */

import { describe, expect, it, vi } from 'vitest';

import { EventBus } from '../src/core/events.js';

type Events = {
  hit: { power: number };
  quiet: undefined;
};

describe('EventBus dispatch', () => {
  it('calls every listener with the payload', () => {
    const bus = new EventBus<Events>();
    const a = vi.fn();
    const b = vi.fn();
    bus.on('hit', a);
    bus.on('hit', b);

    bus.emit('hit', { power: 7 });

    expect(a).toHaveBeenCalledWith({ power: 7 });
    expect(b).toHaveBeenCalledWith({ power: 7 });
  });

  it('routes by type: a listener for one event never sees another', () => {
    const bus = new EventBus<Events>();
    const hit = vi.fn();
    const quiet = vi.fn();
    bus.on('hit', hit);
    bus.on('quiet', quiet);

    bus.emit('quiet', undefined);

    expect(hit).not.toHaveBeenCalled();
    expect(quiet).toHaveBeenCalledTimes(1);
  });

  it('emitting an event with no listeners is a no-op, not a throw', () => {
    const bus = new EventBus<Events>();
    expect(() => bus.emit('hit', { power: 1 })).not.toThrow();
  });

  it('registers the same listener for one type only once', () => {
    const bus = new EventBus<Events>();
    const fn = vi.fn();
    bus.on('hit', fn);
    bus.on('hit', fn);

    bus.emit('hit', { power: 1 });

    // Set semantics: a double-subscribe must not double-fire.
    expect(fn).toHaveBeenCalledTimes(1);
    expect(bus.listenerCount).toBe(1);
  });
});

describe('unsubscribing', () => {
  it('on() returns a disposer that stops delivery', () => {
    const bus = new EventBus<Events>();
    const fn = vi.fn();
    const off = bus.on('hit', fn);

    off();
    bus.emit('hit', { power: 1 });

    expect(fn).not.toHaveBeenCalled();
    expect(bus.listenerCount).toBe(0);
  });

  it('off() removes only the named listener', () => {
    const bus = new EventBus<Events>();
    const keep = vi.fn();
    const drop = vi.fn();
    bus.on('hit', keep);
    bus.on('hit', drop);

    bus.off('hit', drop);
    bus.emit('hit', { power: 1 });

    expect(keep).toHaveBeenCalledTimes(1);
    expect(drop).not.toHaveBeenCalled();
  });

  it('a listener that unsubscribes during dispatch does not skip its neighbours', () => {
    // The failure mode this guards: iterating the live Set while a listener
    // deletes from it can drop the next entry. emit() copies first.
    const bus = new EventBus<Events>();
    const seen: string[] = [];
    const offFirst = bus.on('hit', () => {
      seen.push('first');
      offFirst(); // unsubscribe self, mid-dispatch
    });
    bus.on('hit', () => seen.push('second'));
    bus.on('hit', () => seen.push('third'));

    bus.emit('hit', { power: 1 });

    expect(seen).toEqual(['first', 'second', 'third']);
    expect(bus.listenerCount).toBe(2);
  });

  it('a listener added during dispatch does not fire for the in-flight event', () => {
    const bus = new EventBus<Events>();
    const late = vi.fn();
    bus.on('hit', () => bus.on('hit', late));

    bus.emit('hit', { power: 1 });

    expect(late).not.toHaveBeenCalled();
    bus.emit('hit', { power: 2 });
    expect(late).toHaveBeenCalledTimes(1);
  });

  it('clear() drops one type or everything', () => {
    const bus = new EventBus<Events>();
    bus.on('hit', vi.fn());
    bus.on('quiet', vi.fn());

    bus.clear('hit');
    expect(bus.listenerCount).toBe(1);

    bus.clear();
    expect(bus.listenerCount).toBe(0);
  });
});

describe('once()', () => {
  it('fires exactly one time even when the event repeats', () => {
    const bus = new EventBus<Events>();
    const fn = vi.fn();
    bus.once('hit', fn);

    bus.emit('hit', { power: 1 });
    bus.emit('hit', { power: 2 });
    bus.emit('hit', { power: 3 });

    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledWith({ power: 1 });
    expect(bus.listenerCount).toBe(0);
  });

  it('its returned disposer cancels before the first fire', () => {
    const bus = new EventBus<Events>();
    const fn = vi.fn();
    const off = bus.once('hit', fn);

    off();
    bus.emit('hit', { power: 1 });

    expect(fn).not.toHaveBeenCalled();
  });
});

describe('asEmitter() is publish-only', () => {
  it('delivers to subscribers on the underlying bus', () => {
    const bus = new EventBus<Events>();
    const fn = vi.fn();
    bus.on('hit', fn);

    bus.asEmitter().emit('hit', { power: 4 });

    expect(fn).toHaveBeenCalledWith({ power: 4 });
  });

  it('exposes emit and nothing that can subscribe or clear', () => {
    // A subsystem handed an emitter must not be able to reach back into the
    // bus; that is the decoupling the type exists to enforce.
    const emitter = new EventBus<Events>().asEmitter();
    expect(Object.keys(emitter)).toEqual(['emit']);
    expect(typeof emitter.emit).toBe('function');
    expect((emitter as unknown as Record<string, unknown>).on).toBeUndefined();
    expect((emitter as unknown as Record<string, unknown>).clear).toBeUndefined();
  });
});
