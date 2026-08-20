/**
 * The microphone capture worklet.
 *
 * It runs on the audio thread in the browser, where nothing can be observed,
 * so this is the only place its arithmetic gets checked. Two things matter:
 * the frame is exactly 512 samples because that is the server's VAD window,
 * and float32 -> PCM16 has an asymmetric range that is easy to get wrong in a
 * way you only hear as a crackle.
 */
import { beforeAll, describe, expect, it, vi } from "vitest";

const FRAME = 512;

class BaseProcessor {
  readonly posted: Int16Array[] = [];
  readonly port = {
    postMessage: (msg: Int16Array, transfer?: ArrayBuffer[]) => {
      // Copy first: this is what the other side receives.
      this.posted.push(Int16Array.from(msg));
      // ...and then the sender genuinely loses the buffer, the way a real
      // transferring postMessage detaches it.
      if (transfer) structuredClone(msg, { transfer });
    },
  };
}

type Processor = BaseProcessor & { process(inputs: Float32Array[][]): boolean };

let Capture: new () => Processor;

beforeAll(async () => {
  vi.stubGlobal("AudioWorkletProcessor", BaseProcessor);
  vi.stubGlobal("registerProcessor", (_name: string, cls: new () => Processor) => {
    Capture = cls;
  });
  await import("../public/capture-worklet.js");
});

/** One block of microphone input, as the audio thread delivers it. */
function block(samples: number, fill: number): Float32Array[][] {
  return [[new Float32Array(samples).fill(fill)]];
}

describe("capture worklet", () => {
  it("emits frames of exactly the size the server's VAD expects", () => {
    const proc = new Capture();
    proc.process(block(FRAME, 0.5));
    expect(proc.posted).toHaveLength(1);
    expect(proc.posted[0]).toHaveLength(FRAME);
  });

  it("holds a partial frame back and completes it from the next block", () => {
    const proc = new Capture();
    proc.process(block(300, 0.5));
    expect(proc.posted).toHaveLength(0); // 300 samples is not a frame

    proc.process(block(300, 0.5));
    expect(proc.posted).toHaveLength(1); // 512 of the 600 went out
    proc.process(block(300, 0.5));
    expect(proc.posted).toHaveLength(1); // 88 + 300 is still short
    proc.process(block(300, 0.5));
    expect(proc.posted).toHaveLength(2);
  });

  it("survives its own buffer being transferred away", () => {
    const proc = new Capture();
    proc.process(block(FRAME, 0.5));
    proc.process(block(FRAME, -0.5));

    // Posting the working buffer instead of a copy detaches it, and every
    // frame after the first arrives silent.
    expect(proc.posted[0][0]).toBeGreaterThan(0);
    expect(proc.posted[1][0]).toBeLessThan(0);
  });

  it("uses the full negative rail without wrapping the positive one", () => {
    const proc = new Capture();
    const input = new Float32Array(FRAME);
    input[0] = 1;
    input[1] = -1;
    input[2] = 2; // out of range: a loud mic, or a badly gained one
    input[3] = -2;
    input[4] = 0;
    proc.process([[input]]);

    const out = proc.posted[0];
    // 1.0 * 0x8000 would be 32768, which wraps to -32768: full volume read as
    // full volume the other way round, which is the crackle
    expect(out[0]).toBe(32767);
    expect(out[1]).toBe(-32768);
    expect(out[2]).toBe(32767);
    expect(out[3]).toBe(-32768);
    expect(out[4]).toBe(0);
  });

  it("stays alive when there is no input yet", () => {
    const proc = new Capture();
    expect(proc.process([[]])).toBe(true);
    expect(proc.posted).toHaveLength(0);
  });
});
