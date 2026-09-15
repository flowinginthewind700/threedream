/**
 * Byte-level digests, for every layer that has to prove "these two runs
 * produced the same numbers".
 *
 * This lives in `core/` because both the physics reference scene and the GPU
 * particle layer need it, and `core/` is the only layer both may import: a
 * digest helper that lived in `physics/` would make `gpu/` depend on `physics/`
 * for no reason other than convenience.
 *
 * # Why bytes and not values
 *
 * Comparing floats with `===` after a run tells you the runs agreed; it does not
 * tell you *how much* of the run was compared, and a truncated loop hashes to a
 * match just as happily. So the digest is taken over the raw IEEE-754 bytes of
 * the whole buffer, and reported together with the element count. A digest that
 * does not say what it covered makes a truncation bug look like a pass.
 *
 * # Why two accumulators
 *
 * Plain FNV-1a is order-sensitive but length-insensitive in one annoying way:
 * appending a run of zero bytes to a buffer whose hash already mixed them can
 * collide with a shorter buffer under a naive single-accumulator scheme. The
 * second accumulator folds in the byte position, so a reordering of two
 * equal-magnitude values cannot hash to the same string either.
 */

/**
 * FNV-1a over raw bytes, mixed with a second position-dependent accumulator.
 * Returned as 16 lowercase hex characters.
 */
export function hashBytes(bytes: Uint8Array): string {
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (let i = 0; i < bytes.length; i++) {
    h1 ^= bytes[i]!;
    h1 = Math.imul(h1, 0x01000193) >>> 0;
    h2 = (Math.imul(h2 ^ bytes[i]!, 0x85ebca6b) + (i & 0xff)) >>> 0;
  }
  return `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;
}

/** The digest hex of any typed array, over exactly the bytes it owns. */
export function digestHex(values: ArrayBufferView): string {
  return hashBytes(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
}

/**
 * `hex:count`, the shape every golden digest in this repo uses.
 *
 * `count` is the number of *elements*, not bytes, because that is the number a
 * reader compares against the scene size -- 90649 values from 600 steps is a
 * fact about the run, and its absence is how a truncated run goes unnoticed.
 */
export function digestWithCount(values: ArrayBufferView & { readonly length: number }): string {
  return `${digestHex(values)}:${values.length}`;
}
