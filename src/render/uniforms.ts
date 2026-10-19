import { uniform } from 'three/tsl';
import { Color, Vector2, Vector3 } from 'three/webgpu';
import type { OceanParams } from '../sim/params';

/** Lighting / atmosphere uniforms shared by the sky and the ocean materials. */
export const lighting = {
  sunDir: uniform(new Vector3(0, 1, 0)),
  sunColor: uniform(new Color(1, 1, 1)),
  overcast: uniform(0),
  /** Fog extinction per metre. */
  fogDensity: uniform(0.0002),
  /** World-space XZ of the ocean grid centre (follows the camera). */
  gridOrigin: uniform(new Vector2()),
};

const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

export function updateLighting(p: OceanParams): void {
  const el = (p.sunElevation * Math.PI) / 180;
  const az = (p.sunAzimuth * Math.PI) / 180;
  lighting.sunDir.value.set(Math.cos(az) * Math.cos(el), Math.sin(el), Math.sin(az) * Math.cos(el)).normalize();
  // Sun gets warmer and dimmer as it approaches the horizon (longer optical path through air).
  const day = smoothstep(0, 30, p.sunElevation);
  const intensity = (0.6 + 1.6 * day) * (1 - 0.65 * p.overcast);
  lighting.sunColor.value.setRGB(1.0, 0.5 + 0.45 * day, 0.25 + 0.63 * day).multiplyScalar(intensity);
  lighting.overcast.value = p.overcast;
  lighting.fogDensity.value = p.fogDensity / 1000;
}
