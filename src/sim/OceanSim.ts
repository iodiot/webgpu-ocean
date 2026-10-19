import {
  Fn, cos, exp, float, instanceIndex, int, length, max, mix, pow, saturate, select, sin, sqrt, textureStore,
  uniform, uint, uvec2, vec2, vec3, vec4, instancedArray,
} from 'three/tsl';
import {
  HalfFloatType, LinearFilter, LinearMipmapLinearFilter, RGBAFormat, RepeatWrapping, StorageTexture, Vector2,
  type ComputeNode, type Node, type StorageBufferNode, type WebGPURenderer,
} from 'three/webgpu';
import { cmul, createFFTPass } from './fft';
import { gaussianPairs } from './noise';
import { CASCADES, CASCADE_SIZES, FFT_SIZE, cascadeBands, type OceanParams } from './params';

const N = FFT_SIZE;
const NN = N * N;
const G = 9.81;

/** Pack two complex spectra A and B into A + i·B so one complex IFFT yields two real fields. */
const pack = (a: Node<'vec2'>, b: Node<'vec2'>) => vec2(a.x.sub(b.y), a.y.add(b.x));

/** Shared uniforms: spectrum shape (re-baked on change) and per-frame animation values. */
const u = {
  windDir: uniform(new Vector2(1, 0)),
  windSpeed: uniform(9),
  fetchMeters: uniform(120_000),
  waveScale: uniform(1),
  spread: uniform(0.2),
  spectrumType: uniform(0, 'int'), // 0 = JONSWAP, 1 = Phillips
  time: uniform(0),
  dt: uniform(0),
  choppiness: uniform(1),
  foamAmount: uniform(0.5),
  foamDecay: uniform(1),
};

/** Wave vector k (rad/m) of texel (x, y) in a patch of size L: k = 2π·(n − N/2)/L. */
function waveVector(x: Node<'uint'>, y: Node<'uint'>, L: number): Node<'vec2'> {
  return vec2(float(x), float(y)).sub(N / 2).mul((2 * Math.PI) / L);
}

/**
 * Directional wave-energy density S(kx, kz) in m⁴ (per unit k-area), built from a 1-D frequency
 * spectrum, the deep-water dispersion relation ω = √(gk) and a cos² spreading function.
 */
function spectrumDensity(k: Node<'vec2'>): Node<'float'> {
  const kl = max(length(k), 1e-4);
  const omega = sqrt(kl.mul(G));
  const dOmegaDk = float(G / 2).div(omega);
  const U = max(u.windSpeed, 0.5);

  // JONSWAP (Hasselmann et al. 1973): fetch-limited wind sea, peak-enhancement γ = 3.3.
  const alpha = pow(U.mul(U).div(u.fetchMeters.mul(G)), 0.22).mul(0.076);
  const omegaP = pow(float(G * G).div(U.mul(u.fetchMeters)), 1 / 3).mul(22);
  const sigma = select(omega.lessThanEqual(omegaP), float(0.07), float(0.09));
  const r = exp(omega.sub(omegaP).pow(2).div(sigma.mul(sigma).mul(omegaP).mul(omegaP).mul(2)).negate());
  const jonswapW = alpha.mul(G * G).div(omega.pow(5))
    .mul(exp(omegaP.div(omega).pow(4).mul(-1.25)))
    .mul(pow(3.3, r));
  const jonswapK = jonswapW.mul(dOmegaDk).div(kl); // S(ω) dω → S(k) k dk dθ

  // Phillips (Tessendorf 2001): saturation range α/2·k⁻⁴ with a large-wave cutoff at L = U²/g.
  const Lw = U.mul(U).div(G);
  const phillipsK = float(0.5 * 0.0081).div(kl.pow(4)).mul(exp(float(-1).div(kl.mul(Lw).pow(2))));

  const S = select(u.spectrumType.equal(int(0)), jonswapK, phillipsK);

  // Directional spreading: (2/π)·cos²Δθ on the downwind half-plane, blended toward isotropic.
  const c = k.div(kl).dot(u.windDir);
  const downwind = select(c.greaterThan(0), c.mul(c).mul(2 / Math.PI), float(0));
  const D = mix(downwind, float(1 / (2 * Math.PI)), u.spread);
  return S.mul(D);
}

export class OceanSim {
  /** Per cascade: rgb = displacement (x, y, z) in metres, a = foam. Sampled by the vertex + fragment shaders. */
  readonly displacement: StorageTexture[] = [];
  /** Per cascade: (∂h/∂x, ∂h/∂z, λ∂Dx/∂x, λ∂Dz/∂z): slopes + Jacobian terms for normals. */
  readonly derivatives: StorageTexture[] = [];

  private readonly initPasses: ComputeNode[] = [];
  private readonly framePasses: ComputeNode[] = [];
  private spectrumDirty = true;
  /** Raw buffers of each cascade, kept for the `?debug` self-test (GPU readback vs. a CPU reference). */
  readonly debugBuffers: Array<{ h0: StorageBufferNode<'vec4'>; spectrum: StorageBufferNode<'vec4'> }> = [];

  constructor(private readonly renderer: WebGPURenderer) {
    const bands = cascadeBands();
    for (let c = 0; c < CASCADES; c++) {
      const L = CASCADE_SIZES[c]!;
      const [kMin, kMax] = bands[c]!;
      const disp = makeTexture();
      const deriv = makeTexture();
      this.displacement.push(disp);
      this.derivatives.push(deriv);

      const noise = instancedArray(gaussianPairs(NN, 1337 + c * 7919), 'vec2');
      const h0 = instancedArray(NN, 'vec4'); // (h0(k), conj h0(−k))
      const spectrum = instancedArray(2 * NN, 'vec4'); // 2 grids × 2 packed complex fields = 8 real fields
      const foam = instancedArray(NN, 'float');

      this.debugBuffers.push({ h0, spectrum });
      this.initPasses.push(createInitPass(L, kMin, kMax, noise, h0));
      this.framePasses.push(
        createEvolvePass(L, h0, spectrum),
        createFFTPass(spectrum, 2, false),
        createFFTPass(spectrum, 2, true),
        createAssemblePass(spectrum, foam, disp, deriv),
      );
    }
  }

  /** Push GUI values into uniforms. Spectrum-shaping changes trigger a (cheap, GPU-side) re-bake of h0. */
  setParams(p: OceanParams, spectrumChanged: boolean): void {
    const a = (p.windDirection * Math.PI) / 180;
    u.windDir.value.set(Math.cos(a), Math.sin(a));
    u.windSpeed.value = p.windSpeed;
    u.fetchMeters.value = p.fetch * 1000;
    u.waveScale.value = p.waveScale;
    u.spread.value = p.spread;
    u.spectrumType.value = p.spectrum === 'JONSWAP' ? 0 : 1;
    u.choppiness.value = p.choppiness;
    u.foamAmount.value = p.foamAmount;
    u.foamDecay.value = p.foamDecay;
    if (spectrumChanged) this.spectrumDirty = true;
  }

  /** Current simulation time fed to the shaders (seconds). */
  get time(): number { return u.time.value; }

  /** Advance the simulation; all passes of a frame are recorded into a single compute pass. */
  update(time: number, dt: number): void {
    u.time.value = time;
    u.dt.value = dt;
    const passes = this.spectrumDirty ? [...this.initPasses, ...this.framePasses] : this.framePasses;
    this.spectrumDirty = false;
    this.renderer.compute(passes);
  }
}

function makeTexture(): StorageTexture {
  const t = new StorageTexture(N, N);
  t.type = HalfFloatType; // rgba16float: storage-writable *and* filterable in core WebGPU
  t.format = RGBAFormat;
  t.wrapS = t.wrapT = RepeatWrapping;
  t.magFilter = LinearFilter;
  t.minFilter = LinearMipmapLinearFilter; // three regenerates the mip chain after each compute write
  t.generateMipmaps = true;
  t.anisotropy = 8;
  return t;
}

/** h0(k) = (ξr + iξi)·√(S(k)Δk²)/2 (Tessendorf eq. 42, with Δk-correct normalisation). */
function createInitPass(L: number, kMin: number, kMax: number, noise: StorageBufferNode<'vec2'>, h0: StorageBufferNode<'vec4'>) {
  return Fn(() => {
    const i = instanceIndex;
    const x = i.bitAnd(N - 1);
    const y = i.shiftRight(8);
    // Index of −k (mod N): needed so h(k,t) and h(−k,t) stay complex conjugates → real output.
    const xm = uint(N).sub(x).bitAnd(N - 1);
    const ym = uint(N).sub(y).bitAnd(N - 1);
    const iNeg = ym.mul(N).add(xm);

    const dk = (2 * Math.PI) / L;
    const amplitude = (k: Node<'vec2'>) => {
      const kl = length(k);
      // Each cascade only owns its band of wavenumbers, so no wave is counted twice.
      const inBand = kl.greaterThanEqual(kMin).and(kl.lessThan(kMax)).and(kl.greaterThan(1e-4));
      return select(inBand, sqrt(spectrumDensity(k).mul(dk * dk)).mul(0.5).mul(u.waveScale), float(0));
    };
    const k = waveVector(x, y, L);
    const hk = noise.element(i).mul(amplitude(k));
    const hmk = noise.element(iNeg).mul(amplitude(k.negate()));
    // The Nyquist row/column (n = −N/2) has no conjugate partner on the grid: zero it.
    const keep = select(x.equal(0).or(y.equal(0)), float(0), float(1));
    h0.element(i).assign(vec4(hk.x, hk.y, hmk.x, hmk.y.negate()).mul(keep));
  })().compute(NN);
}

/**
 * Time evolution h(k,t) = h0(k)e^{−iωt} + conj(h0(−k))e^{iωt}, then the 8 spectra we need
 * (height, horizontal displacement, slopes, Jacobian terms), packed pairwise into 4 complex fields.
 */
function createEvolvePass(L: number, h0: StorageBufferNode<'vec4'>, spectrum: StorageBufferNode<'vec4'>) {
  return Fn(() => {
    const i = instanceIndex;
    const k = waveVector(i.bitAnd(N - 1), i.shiftRight(8), L).toVar();
    const kl = max(length(k), 1e-4).toVar();
    const phase = sqrt(kl.mul(G)).mul(u.time);
    const e = vec2(cos(phase), sin(phase)).toVar();
    const h0v = h0.element(i).toVar();
    const h = cmul(h0v.xy, vec2(e.x, e.y.negate())).add(cmul(h0v.zw, e)).toVar();
    const ih = vec2(h.y.negate(), h.x).toVar(); // i·h

    const dx = ih.mul(k.x.div(kl)); //  i kx/|k| h  → horizontal displacement X
    const dz = ih.mul(k.y.div(kl)); //  i kz/|k| h  → horizontal displacement Z
    const sx = ih.mul(k.x); //          i kx h      → ∂h/∂x
    const sz = ih.mul(k.y); //          i kz h      → ∂h/∂z
    const dxx = h.mul(k.x.mul(k.x).div(kl).negate()); // ∂Dx/∂x
    const dzz = h.mul(k.y.mul(k.y).div(kl).negate()); // ∂Dz/∂z
    const dxz = h.mul(k.x.mul(k.y).div(kl).negate()); // ∂Dx/∂z

    spectrum.element(i).assign(vec4(pack(dx, h), pack(dz, dxz)));
    spectrum.element(i.add(NN)).assign(vec4(pack(sx, sz), pack(dxx, dzz)));
  })().compute(NN);
}

/**
 * After the 2-D IFFT: undo the (−1)^(x+y) shift from centring k, apply choppiness, compute the
 * Jacobian of the horizontal displacement (J < 1 where the surface folds → whitecaps), accumulate
 * foam with exponential decay, and write the two textures the renderer samples.
 */
function createAssemblePass(spectrum: StorageBufferNode<'vec4'>, foam: StorageBufferNode<'float'>, disp: StorageTexture, deriv: StorageTexture) {
  return Fn(() => {
    const i = instanceIndex;
    const x = i.bitAnd(N - 1);
    const y = i.shiftRight(8);
    const sign = select(x.add(y).bitAnd(1).equal(0), float(1), float(-1));
    const a = spectrum.element(i).mul(sign).toVar(); //       (Dx, Dy, Dz, ∂Dx/∂z)
    const b = spectrum.element(i.add(NN)).mul(sign).toVar(); // (∂h/∂x, ∂h/∂z, ∂Dx/∂x, ∂Dz/∂z)
    const lambda = u.choppiness;

    const jxx = b.z.mul(lambda);
    const jzz = b.w.mul(lambda);
    const jxz = a.w.mul(lambda);
    const jacobian = jxx.add(1).mul(jzz.add(1)).sub(jxz.mul(jxz));

    const threshold = u.foamAmount.mul(0.8).add(0.1);
    const inject = saturate(threshold.sub(jacobian).div(0.4));
    const f = max(foam.element(i).mul(exp(u.foamDecay.mul(u.dt).negate())), inject).toVar();
    foam.element(i).assign(f);

    const texel = uvec2(x, y);
    textureStore(disp, texel, vec4(vec3(a.x.mul(lambda), a.y, a.z.mul(lambda)), f));
    textureStore(deriv, texel, vec4(b.x, b.y, jxx, jzz));
  })().compute(NN);
}
