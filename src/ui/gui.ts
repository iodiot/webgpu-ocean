import GUI from 'lil-gui';
import { PRESETS, SPECTRUM_KEYS, type OceanParams, type PresetName } from '../sim/params';

export interface GuiState { preset: PresetName | 'Custom' }

/**
 * lil-gui panel bound to the live params object.
 * `onChange(spectrumChanged)` tells the caller whether the initial spectrum h0 must be re-baked.
 */
export function createGui(params: OceanParams, state: GuiState, onChange: (spectrumChanged: boolean) => void): GUI {
  const gui = new GUI({ title: 'Ocean' });
  const spectrumKeys = new Set<string>(SPECTRUM_KEYS);

  gui.add(state, 'preset', ['Calm', 'Breeze', 'Storm', 'Custom']).name('sea state').onChange((name: string) => {
    if (name === 'Custom') return;
    Object.assign(params, PRESETS[name as PresetName]);
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
    onChange(true);
  });

  // Any manual tweak turns the preset label into "Custom".
  const bind = (folder: GUI, key: keyof OceanParams, min?: number, max?: number, step?: number, label?: string) => {
    const c = min === undefined ? folder.add(params, key) : folder.add(params, key, min, max, step);
    if (label) c.name(label);
    c.onChange(() => {
      if (state.preset !== 'Custom' && key !== 'paused' && key !== 'timeScale') {
        state.preset = 'Custom';
        gui.controllersRecursive().forEach((ctl) => ctl.updateDisplay());
      }
      onChange(spectrumKeys.has(key));
    });
    return c;
  };

  const w = gui.addFolder('Waves');
  w.add(params, 'spectrum', ['JONSWAP', 'Phillips']).onChange(() => onChange(true));
  bind(w, 'windSpeed', 1, 30, 0.1, 'wind speed (m/s)');
  bind(w, 'windDirection', 0, 360, 1, 'wind direction (°)');
  bind(w, 'fetch', 5, 1000, 1, 'fetch (km)');
  bind(w, 'waveScale', 0, 2, 0.01, 'wave scale');
  bind(w, 'choppiness', 0, 2, 0.01);
  bind(w, 'spread', 0, 1, 0.01, 'directional spread');

  const f = gui.addFolder('Foam');
  bind(f, 'foamAmount', 0, 1, 0.01, 'amount');
  bind(f, 'foamDecay', 0.1, 4, 0.05, 'decay (1/s)');

  const l = gui.addFolder('Lighting');
  bind(l, 'sunElevation', -2, 85, 0.5, 'sun elevation (°)');
  bind(l, 'sunAzimuth', 0, 360, 1, 'sun azimuth (°)');
  bind(l, 'overcast', 0, 1, 0.01);
  bind(l, 'exposure', 0.2, 2.5, 0.01);
  bind(l, 'fogDensity', 0, 2, 0.01, 'fog (per km)');

  const t = gui.addFolder('Time');
  bind(t, 'timeScale', 0, 3, 0.05, 'time scale');
  bind(t, 'paused', undefined, undefined, undefined, 'pause (space)');
  return gui;
}
