import { dot, max, mix, normalize, positionWorld, pow, cameraPosition, saturate, smoothstep, vec3 } from 'three/tsl';
import { BackSide, Mesh, MeshBasicNodeMaterial, SphereGeometry, type Node } from 'three/webgpu';
import { lighting } from './uniforms';

/**
 * Cheap analytic sky: horizon→zenith gradient that warms at sunset, a two-lobe sun halo,
 * optional sun disk and an overcast blend. Used for the sky dome, the ocean's reflections
 * and the fog colour, so all three always agree (no seam at the horizon).
 */
export function skyColor(dir: Node<'vec3'>, withSunDisk: boolean): Node<'vec3'> {
  const { sunDir, sunColor, overcast } = lighting;
  const y = max(dir.y, 0); // below the horizon the "sky" is the horizon colour (= fog colour)
  const day = smoothstep(-0.05, 0.3, sunDir.y);

  const zenith = vec3(0.045, 0.15, 0.42).mul(mix(0.2, 1, day));
  const horizon = mix(vec3(0.95, 0.52, 0.3), vec3(0.42, 0.56, 0.7), smoothstep(0.0, 0.35, sunDir.y)).mul(mix(0.3, 1, day));
  let col = mix(horizon, zenith, pow(y, 0.45));

  const mu = saturate(dot(dir, sunDir));
  const halo = sunColor.mul(pow(mu, 8).mul(0.12).add(pow(mu, 256).mul(0.25)));
  col = col.add(halo.mul(overcast.mul(-0.7).add(1)));

  const grey = vec3(0.4, 0.44, 0.48).mul(mix(0.25, 1, day)).mul(y.mul(0.25).add(0.8));
  col = mix(col, grey, overcast.mul(0.85));

  if (withSunDisk) {
    const disk = smoothstep(0.99996, 0.99999, mu).mul(40).mul(overcast.oneMinus().pow(2));
    col = col.add(sunColor.mul(disk));
  }
  return col;
}

/** Inside-out sphere that follows the camera. */
export function createSkyDome(radius: number): Mesh {
  const material = new MeshBasicNodeMaterial({ side: BackSide, depthWrite: false });
  material.colorNode = skyColor(normalize(positionWorld.sub(cameraPosition)), true);
  material.fog = false;
  const mesh = new Mesh(new SphereGeometry(radius, 48, 24), material);
  mesh.frustumCulled = false;
  mesh.renderOrder = -1;
  mesh.onBeforeRender = (_r, _s, camera) => mesh.position.copy(camera.position);
  return mesh;
}
