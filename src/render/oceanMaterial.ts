import {
  Fn, abs, attribute, cameraPosition, dot, exp, float, length, log2, max, mix, normalize, pow, reflect, saturate,
  smoothstep, texture, varying, vec2, vec3,
} from 'three/tsl';
import { DoubleSide, MeshBasicNodeMaterial, type StorageTexture } from 'three/webgpu';
import { CASCADE_SIZES, FFT_SIZE } from '../sim/params';
import { skyColor } from './sky';
import { lighting } from './uniforms';

/** Foam contribution per cascade: big breaking waves matter, sub-metre ripples barely do. */
const FOAM_WEIGHTS = [1.0, 0.35, 0.1];

const DEEP_COLOR = vec3(0.002, 0.016, 0.026);
const SCATTER_COLOR = vec3(0.02, 0.17, 0.14);
const FOAM_COLOR = vec3(0.86, 0.9, 0.92);

/**
 * Ocean surface material (TSL). Unlit `MeshBasicNodeMaterial`: all shading is custom, and three's
 * output stage applies ACES tone mapping + sRGB.
 *
 * Vertex: displace the polar grid by the sum of the three cascades' (x, y, z) displacement,
 *         choosing a texture LOD from the local vertex spacing to avoid geometric aliasing.
 * Fragment: per-pixel normals from slopes + Jacobian terms, Fresnel-weighted sky reflection,
 *         GGX sun glint, a fake subsurface-scattering term on backlit crests, foam, fog.
 */
export function createOceanMaterial(displacement: StorageTexture[], derivatives: StorageTexture[], gridSpacing: number, gridRadius: number): MeshBasicNodeMaterial {
  const { sunDir, sunColor, overcast, fogDensity, gridOrigin } = lighting;
  const material = new MeshBasicNodeMaterial({ side: DoubleSide });
  material.fog = false;

  // ---------------- vertex ----------------
  // The grid mesh sits at the origin; translating it here keeps it centred on the viewer.
  // (Raw attribute rather than positionLocal, which three overwrites with positionNode.)
  const baseXZ = attribute<'vec3'>('position', 'vec3').xz.add(gridOrigin);
  const vertexDisplacement = Fn(() => {
    const dist = length(baseXZ.sub(cameraPosition.xz));
    const spacing = max(dist.mul(gridSpacing), 0.01); // metres between neighbouring vertices
    const d = vec3(0).toVar();
    CASCADE_SIZES.forEach((L, c) => {
      const lod = log2(max(spacing.div(L / FFT_SIZE), 1));
      d.addAssign(texture(displacement[c]!, baseXZ.div(L)).level(lod).xyz);
    });
    return vec3(baseXZ.x.add(d.x), d.y, baseXZ.y.add(d.z));
  })();
  material.positionNode = vertexDisplacement;

  // Undisplaced (Lagrangian) XZ: the textures are indexed by where a water particle *rests*.
  const vBase = varying(baseXZ);
  const vWorld = varying(vertexDisplacement);

  // ---------------- fragment ----------------
  material.colorNode = Fn(() => {
    // Sum slopes / Jacobian terms / foam over cascades (hardware mips + anisotropic filtering).
    const deriv = vec3(0).toVar();
    const jac = vec2(0).toVar();
    const foam = float(0).toVar();
    CASCADE_SIZES.forEach((L, c) => {
      const uv = vBase.div(L);
      const s = texture(derivatives[c]!, uv);
      deriv.addAssign(vec3(s.x, s.y, 0));
      jac.addAssign(s.zw);
      foam.addAssign(texture(displacement[c]!, uv).w.mul(FOAM_WEIGHTS[c] ?? 0));
    });

    // Normal of the *displaced* surface: slope divided by the horizontal stretch (1 + λ∂D/∂x).
    const n = normalize(vec3(
      deriv.x.negate().div(max(jac.x.add(1), 0.2)),
      1,
      deriv.y.negate().div(max(jac.y.add(1), 0.2)),
    )).toVar();

    const toEye = cameraPosition.sub(vWorld);
    const dist = length(toEye).toVar();
    const v = toEye.div(dist).toVar();
    const l = sunDir;
    const NdotV = max(dot(n, v), 0.001);
    const NdotL = saturate(dot(n, l));

    // Fresnel (Schlick, water F0 ≈ 0.02) and reflected sky. Reflection rays that would point into
    // the water (steep, wave-occluded normals) are mirrored back up rather than sampling "below".
    const fresnel = float(0.02).add(pow(NdotV.oneMinus(), 5).mul(0.98)).toVar();
    const r = reflect(v.negate(), n);
    const reflected = skyColor(normalize(vec3(r.x, abs(r.y), r.z)), false);

    // Sun glint: GGX distribution. Roughness grows with distance because the mip chain averages
    // away the small waves that would otherwise spread the highlight.
    const roughness = mix(0.07, 0.3, smoothstep(0, 4000, dist));
    const a2 = roughness.mul(roughness);
    const h = normalize(l.add(v));
    const NdotH = saturate(dot(n, h));
    const denom = NdotH.mul(NdotH).mul(a2.sub(1)).add(1);
    const D = a2.div(denom.mul(denom).mul(Math.PI));
    const Fh = float(0.02).add(pow(saturate(dot(h, v)).oneMinus(), 5).mul(0.98));
    const glint = sunColor.mul(D.mul(Fh).mul(NdotL).div(max(NdotV, 0.15).mul(4)))
      .mul(overcast.oneMinus().pow(2)).mul(smoothstep(-0.02, 0.05, l.y));

    // Water body + "subsurface" scattering (after Atlas / Sea of Thieves style approximations):
    // light leaking through tall crests when looking toward the sun, plus a view-facing term.
    const ambient = skyColor(vec3(0, 1, 0), false);
    const height = max(vWorld.y, 0);
    const backlit = pow(saturate(dot(l, v.negate())), 4).mul(pow(dot(l, n).mul(-0.5).add(0.5), 3));
    const sss = height.mul(backlit).mul(1.6).add(pow(NdotV, 2).mul(0.25)).add(NdotL.mul(0.08));
    const water = DEEP_COLOR.mul(ambient.mul(2)).add(SCATTER_COLOR.mul(sunColor).mul(sss));

    let color = mix(water, reflected, fresnel).add(glint);

    // Whitecaps: diffuse-lit foam on top.
    const foamLight = sunColor.mul(NdotL.mul(0.6).add(0.25)).add(ambient.mul(0.6));
    color = mix(color, FOAM_COLOR.mul(foamLight), saturate(foam).mul(0.92));

    // Exponential distance fog toward the horizon colour; forced to 1 at the grid edge so the
    // finite mesh blends seamlessly into the sky dome (whose below-horizon colour is the same).
    const horizDist = length(vWorld.xz.sub(cameraPosition.xz));
    const fogAmount = max(exp(dist.mul(fogDensity).negate()).oneMinus(), smoothstep(gridRadius * 0.5, gridRadius * 0.92, horizDist));
    const fogColor = skyColor(normalize(vec3(v.x.negate(), 0, v.z.negate())), false);
    return mix(color, fogColor, fogAmount);
  })();

  return material;
}
