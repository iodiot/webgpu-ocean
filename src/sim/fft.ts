import { Fn, cos, float, localId, sin, vec2, vec4, workgroupArray, workgroupBarrier, workgroupId } from 'three/tsl';
import type { Node, StorageBufferNode } from 'three/webgpu';
import { FFT_LOG2, FFT_SIZE } from './params';

const N = FFT_SIZE;
const HALF = N / 2;

/** Complex multiply of two vec2 nodes (re, im). */
export const cmul = (a: Node<'vec2'>, b: Node<'vec2'>): Node<'vec2'> =>
  vec2(a.x.mul(b.x).sub(a.y.mul(b.y)), a.x.mul(b.y).add(a.y.mul(b.x)));

/**
 * In-place inverse 1-D FFT of every row (or column) of a stack of N×N complex grids.
 *
 * Layout: `data` holds `groups` grids of N×N `vec4`s; each vec4 is *two* complex numbers
 * (xy, zw) that are transformed together, so one pass handles 2 signals.
 *
 * One workgroup owns one line: it loads the line into workgroup memory, runs log2(N) radix-2
 * Stockham stages (no bit reversal needed, the output is in natural order), and writes it back.
 * The stage loop is unrolled in JS, so every twiddle stride is a compile-time constant.
 *
 * Dispatch: [N lines, groups, 1] workgroups of N/2 threads (one butterfly per thread per stage).
 */
export function createFFTPass(data: StorageBufferNode<'vec4'>, groups: number, vertical: boolean) {
  // Two halves of a ping-pong buffer: [0, N) and [N, 2N).
  const shared = workgroupArray('vec4', 2 * N);

  const pass = Fn(() => {
    const t = localId.x; // 0 … N/2-1
    const line = workgroupId.x; // row (horizontal pass) or column (vertical pass)
    const base = workgroupId.y.mul(N * N); // which grid of the stack

    const addr = (i: Node<'uint'>) => (vertical ? base.add(i.mul(N)).add(line) : base.add(line.mul(N)).add(i));
    // (WorkgroupInfoNode.element() exists at runtime but is missing from @types/three.)
    const sharedArr = shared as unknown as { element(i: Node<'uint'>): Node<'vec4'> };
    const at = (i: Node<'uint'>) => sharedArr.element(i);

    at(t).assign(data.element(addr(t)));
    at(t.add(HALF)).assign(data.element(addr(t.add(HALF))));
    workgroupBarrier();

    let src = 0;
    for (let s = 0; s < FFT_LOG2; s++) {
      const p = 1 << s; // length of the sub-transforms already done
      const dst = N - src;
      const k = t.bitAnd(p - 1);
      const u0 = at(t.add(src)).toVar();
      const u1 = at(t.add(src + HALF)).toVar();
      // Inverse transform: twiddle e^{+iπk/p}.
      const angle = float(k).mul(Math.PI / p);
      const w = vec2(cos(angle), sin(angle)).toVar();
      const v = vec4(cmul(u1.xy, w), cmul(u1.zw, w)).toVar();
      const j = t.mul(2).sub(k);
      at(j.add(dst)).assign(u0.add(v));
      at(j.add(dst + p)).assign(u0.sub(v));
      workgroupBarrier();
      src = dst;
    }

    data.element(addr(t)).assign(at(t.add(src)));
    data.element(addr(t.add(HALF))).assign(at(t.add(src + HALF)));
  })().computeKernel([HALF]);
  pass.dispatchSize = [N, groups, 1]; // workgroup counts: one workgroup per line
  return pass;
}
