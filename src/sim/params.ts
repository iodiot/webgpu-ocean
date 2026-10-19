/** Tunables for the ocean simulation and shading, plus the Calm / Breeze / Storm presets. */

/** FFT resolution per cascade. Fixed: the WGSL FFT kernel is specialised for 256 (8 radix-2 stages). */
export const FFT_SIZE = 256;
export const FFT_LOG2 = 8;

/**
 * Three cascades of different physical patch sizes are summed to hide tiling and cover
 * wavelengths from ~250 m swell down to ~5 cm ripples. Sizes are deliberately non-integer
 * multiples so the repeat periods never line up.
 */
export const CASCADE_SIZES = [250, 37, 7.3] as const;
export const CASCADES = CASCADE_SIZES.length;

/**
 * Wavenumber band [kMin, kMax) owned by each cascade, so every wave is simulated exactly once.
 * A smaller patch takes over from 6 of its own frequency steps (6·2π/L): below that it has too few
 * samples to represent the wave well.
 */
export function cascadeBands(): Array<[number, number]> {
  const bands: Array<[number, number]> = [];
  for (let i = 0; i < CASCADES; i++) {
    const kMin = i === 0 ? 0 : (6 * 2 * Math.PI) / CASCADE_SIZES[i]!;
    const kMax = i === CASCADES - 1 ? 1e9 : (6 * 2 * Math.PI) / CASCADE_SIZES[i + 1]!;
    bands.push([kMin, kMax]);
  }
  return bands;
}

export type SpectrumType = 'JONSWAP' | 'Phillips';

export interface OceanParams {
  // --- spectrum (changing these re-generates the initial spectrum h0) ---
  spectrum: SpectrumType;
  /** Wind speed at 10 m above the sea, m/s. */
  windSpeed: number;
  /** Wind direction, degrees (0 = +X). */
  windDirection: number;
  /** Fetch: distance over which the wind has blown, km (JONSWAP only). */
  fetch: number;
  /** Global multiplier on wave amplitude. */
  waveScale: number;
  /** 0 = waves strictly follow the wind (cos²), 1 = fully isotropic sea. */
  spread: number;
  // --- per-frame ---
  /** Horizontal displacement factor λ (Tessendorf "choppiness"). */
  choppiness: number;
  /** 0…1, how easily the Jacobian threshold triggers whitecaps. */
  foamAmount: number;
  /** How quickly foam dissipates, 1/s. */
  foamDecay: number;
  timeScale: number;
  paused: boolean;
  // --- lighting ---
  /** Sun elevation above the horizon, degrees. */
  sunElevation: number;
  /** Sun azimuth, degrees. */
  sunAzimuth: number;
  /** 0 = clear sky, 1 = heavy overcast (greys the sky, dims the sun). */
  overcast: number;
  /** Exposure before ACES tone mapping. */
  exposure: number;
  /** Fog density (per km). */
  fogDensity: number;
}

export type PresetName = 'Calm' | 'Breeze' | 'Storm';

type PresetValues = Pick<OceanParams,
  'windSpeed' | 'fetch' | 'waveScale' | 'spread' | 'choppiness' | 'foamAmount' | 'foamDecay' |
  'sunElevation' | 'overcast' | 'exposure' | 'fogDensity'>;

export const PRESETS: Record<PresetName, PresetValues> = {
  Calm:   { windSpeed: 3.5, fetch: 40,  waveScale: 1.0, spread: 0.25, choppiness: 0.7, foamAmount: 0.15, foamDecay: 1.6,
            sunElevation: 28, overcast: 0, exposure: 1.0, fogDensity: 0.12 },
  Breeze: { windSpeed: 9,   fetch: 120, waveScale: 1.0, spread: 0.2,  choppiness: 1.0, foamAmount: 0.45, foamDecay: 1.0,
            sunElevation: 20, overcast: 0.05, exposure: 1.0, fogDensity: 0.18 },
  Storm:  { windSpeed: 21,  fetch: 400, waveScale: 1.0, spread: 0.12, choppiness: 1.25, foamAmount: 0.65, foamDecay: 0.6,
            sunElevation: 9,  overcast: 0.85, exposure: 0.85, fogDensity: 0.55 },
};

export function defaultParams(): OceanParams {
  return {
    spectrum: 'JONSWAP',
    windDirection: 30,
    timeScale: 1,
    paused: false,
    sunAzimuth: 200,
    ...PRESETS.Breeze,
  };
}

/** Keys that require regenerating h0(k) when they change. */
export const SPECTRUM_KEYS = ['spectrum', 'windSpeed', 'windDirection', 'fetch', 'waveScale', 'spread'] as const;
