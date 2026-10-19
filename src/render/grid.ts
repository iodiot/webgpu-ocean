import { BufferAttribute, BufferGeometry } from 'three/webgpu';

/**
 * Polar "screen-space-ish" LOD grid centred on the viewer: ring radii grow exponentially, so the
 * vertex spacing is roughly proportional to distance (≈ constant projected size), from ~1 cm under
 * the camera out to `radius` metres. ~190k vertices replace what would be millions in a uniform grid.
 */
export function createPolarGrid(radius: number, segments = 384, innerRadius = 0.5): { geometry: BufferGeometry; spacing: number } {
  const ratio = (2 * Math.PI) / segments; // angular spacing; use the same radial ratio → square-ish cells
  const rings = Math.ceil(Math.log(radius / innerRadius) / ratio) + 1;
  const positions = new Float32Array((1 + rings * segments) * 3);
  // Vertex 0 is the centre; ring r, segment s is at 1 + r·segments + s.
  for (let r = 0; r < rings; r++) {
    const rad = Math.min(radius, innerRadius * Math.exp(ratio * r));
    for (let s = 0; s < segments; s++) {
      const a = s * ratio;
      const o = (1 + r * segments + s) * 3;
      positions[o] = rad * Math.cos(a);
      positions[o + 2] = rad * Math.sin(a);
    }
  }
  const indices = new Uint32Array((segments + (rings - 1) * segments * 2) * 3);
  let n = 0;
  for (let s = 0; s < segments; s++) {
    const s1 = (s + 1) % segments;
    indices.set([0, 1 + s1, 1 + s], n);
    n += 3;
  }
  for (let r = 0; r < rings - 1; r++) {
    for (let s = 0; s < segments; s++) {
      const s1 = (s + 1) % segments;
      const a = 1 + r * segments + s, b = 1 + r * segments + s1;
      const c = 1 + (r + 1) * segments + s, d = 1 + (r + 1) * segments + s1;
      indices.set([a, b, c, b, d, c], n);
      n += 6;
    }
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(positions, 3));
  geometry.setIndex(new BufferAttribute(indices, 1));
  return { geometry, spacing: ratio };
}
