import { ACESFilmicToneMapping, Mesh, PerspectiveCamera, Scene, WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createPolarGrid } from './render/grid';
import { createOceanMaterial } from './render/oceanMaterial';
import { createSkyDome } from './render/sky';
import { lighting, updateLighting } from './render/uniforms';
import { OceanSim } from './sim/OceanSim';
import { defaultParams, PRESETS, type PresetName } from './sim/params';
import { createGui, type GuiState } from './ui/gui';
import { Stats } from './ui/stats';

const GRID_RADIUS = 9000; // metres of simulated surface around the viewer
const MIN_EYE_HEIGHT = 2.5;

function showFallback(reason: string): void {
  document.getElementById('fallback')!.style.display = 'flex';
  document.getElementById('fallback-reason')!.textContent = reason;
  document.getElementById('stats')!.style.display = 'none';
  document.getElementById('hint')!.style.display = 'none';
  document.querySelector<HTMLElement>('.lil-gui')?.style.setProperty('display', 'none');
}

/** three.js would silently fall back to WebGL 2, but the FFT needs compute + workgroup memory. */
async function requireWebGPU(): Promise<string> {
  if (!('gpu' in navigator)) throw new Error('navigator.gpu is undefined — this browser does not expose WebGPU.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No suitable GPU adapter was found (WebGPU may be disabled or blocklisted).');
  const info = adapter.info;
  return [info?.vendor, info?.architecture || info?.description].filter(Boolean).join(' ') || 'unknown';
}

async function main(): Promise<void> {
  const adapterInfo = await requireWebGPU();

  const renderer = new WebGPURenderer({ antialias: true, powerPreference: 'high-performance' });
  await renderer.init();
  if (!(renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend) {
    throw new Error('three.js could not initialise its WebGPU backend.');
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.onDeviceLost = (info) => showFallback(`GPU device lost: ${info.message}`);
  renderer.domElement.id = 'gpu';
  document.body.prepend(renderer.domElement);

  const params = defaultParams();
  const state: GuiState = { preset: 'Breeze' };
  const q = new URLSearchParams(location.search).get('preset');
  if (q && q in PRESETS) { state.preset = q as PresetName; Object.assign(params, PRESETS[state.preset]); }

  // ---- simulation ----
  const sim = new OceanSim(renderer);
  const apply = (spectrumChanged: boolean) => {
    sim.setParams(params, spectrumChanged);
    updateLighting(params);
    renderer.toneMappingExposure = params.exposure;
  };
  apply(true);

  // ---- scene ----
  const scene = new Scene();
  const camera = new PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.3, 30_000);
  camera.position.set(48, 14, 22);
  const { geometry, spacing } = createPolarGrid(GRID_RADIUS);
  const ocean = new Mesh(geometry, createOceanMaterial(sim.displacement, sim.derivatives, spacing, GRID_RADIUS));
  ocean.frustumCulled = false; // vertices are moved in the shader; bounds are meaningless
  scene.add(createSkyDome(20_000), ocean);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 2, 0);
  controls.enableDamping = true;
  controls.minDistance = 6;
  controls.maxDistance = 500;
  controls.maxPolarAngle = Math.PI / 2 - 0.01;
  controls.update();

  // ?debug exposes internals for automated checks / screenshot capture (see README → Verification).
  let captureRequest: ((png: string) => void) | null = null;
  if (new URLSearchParams(location.search).has('debug')) {
    const capture = () => new Promise<string>((resolve) => { captureRequest = resolve; });
    Object.assign(window, { __ocean: { renderer, sim, params, apply, camera, controls, capture } });
  }

  const stats = new Stats(document.getElementById('stats')!, adapterInfo);
  createGui(params, state, apply);
  window.addEventListener('keydown', (e) => { if (e.code === 'Space') { params.paused = !params.paused; e.preventDefault(); } });
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  let last = performance.now();
  let simTime = 0;
  renderer.setAnimationLoop((now: number) => {
    const dtMs = now - last;
    last = now;
    // Clamp dt so a backgrounded tab doesn't make foam decay / time jump.
    const dt = params.paused ? 0 : Math.min(dtMs / 1000, 1 / 20) * params.timeScale;
    simTime += dt;

    controls.update();
    if (camera.position.y < MIN_EYE_HEIGHT) camera.position.y = MIN_EYE_HEIGHT; // stay above the swell
    // Keep the LOD grid centred under the camera; snapping avoids sub-vertex "swimming".
    lighting.gridOrigin.value.set(Math.round(camera.position.x * 4) / 4, Math.round(camera.position.z * 4) / 4);

    sim.update(simTime, dt);
    renderer.render(scene, camera);
    // Read the canvas in the same task as the render, before the swap-chain texture expires.
    if (captureRequest) { captureRequest(renderer.domElement.toDataURL('image/png')); captureRequest = null; }
    stats.frame(dtMs, renderer.info.render.triangles, state.preset, params.paused);
  });
}

main().catch((err: unknown) => {
  console.error(err);
  showFallback(err instanceof Error ? err.message : String(err));
});
