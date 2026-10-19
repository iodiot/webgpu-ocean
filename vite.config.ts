import { defineConfig } from 'vite';

// GitHub Pages serves the project at https://<user>.github.io/webgpu-ocean/
export default defineConfig({
  base: '/webgpu-ocean/',
  // three.webgpu is ~900 kB minified (260 kB gzip); it's a single-page demo, so one chunk is fine.
  build: { target: 'es2022', chunkSizeWarningLimit: 1200 },
});
