/** Deterministic Gaussian noise for the initial spectrum, so every visitor sees the same sea. */

/** mulberry32: tiny, fast, good-enough 32-bit PRNG. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** `count` pairs of independent N(0,1) samples (Box–Muller), packed as [g0, g1, g0, g1, ...]. */
export function gaussianPairs(count: number, seed: number): Float32Array {
  const rand = mulberry32(seed);
  const out = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    const u1 = Math.max(rand(), 1e-12);
    const u2 = rand();
    const r = Math.sqrt(-2 * Math.log(u1));
    out[2 * i] = r * Math.cos(2 * Math.PI * u2);
    out[2 * i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  return out;
}
