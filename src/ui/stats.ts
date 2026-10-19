/** Rolling-average stats overlay (FPS, CPU frame time, triangles, adapter). */
export class Stats {
  private frames = 0;
  private acc = 0;
  private lastUpdate = performance.now();

  constructor(private readonly el: HTMLElement, private readonly adapterInfo: string) {}

  frame(dtMs: number, triangles: number, preset: string, paused: boolean): void {
    this.frames++;
    this.acc += dtMs;
    const now = performance.now();
    if (now - this.lastUpdate < 250) return;
    const avg = this.acc / this.frames;
    this.frames = 0;
    this.acc = 0;
    this.lastUpdate = now;
    this.el.innerHTML =
      `<b>WebGPU Ocean</b>${paused ? '  [paused]' : ''}\n` +
      `FPS        ${(1000 / avg).toFixed(0)}\n` +
      `frame      ${avg.toFixed(2)} ms\n` +
      `FFT        3 × 256² cascades\n` +
      `triangles  ${(triangles / 1000).toFixed(0)}k\n` +
      `sea state  ${preset}\n` +
      `adapter    ${this.adapterInfo}`;
  }
}
