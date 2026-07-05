import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import * as esbuild from "esbuild";

const root = path.resolve(import.meta.dirname, "..");
const modules = await buildTestModules();
const { colorCode } = await import(modules.aiSnapshot);

test("GPU snapshot helpers normalize numeric and cased colors", () => {
  assert.equal(colorCode("WHITE"), 0);
  assert.equal(colorCode("BLACK"), 1);
  assert.equal(colorCode(0), 0);
  assert.equal(colorCode(1), 1);
});

test("GPU snapshot color codes delegate to engine when available", () => {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const outputPtr = 32768;
  let outputLen = 0;
  let nextPtr = 1024;
  let calls = 0;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder("utf-8");
  const engine = {
    memory,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_last_message() {
      return outputPtr;
    },
    chronofish_gpu_search_color_code_json(ptr, length) {
      calls += 1;
      assert.equal(decoder.decode(new Uint8Array(memory.buffer, ptr, length)), "BLACK");
      const bytes = encoder.encode("1");
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    }
  };

  assert.equal(colorCode("BLACK", engine), 1);
  assert.equal(calls, 1);
});

async function buildTestModules() {
  const outdir = await mkdtemp(path.join(os.tmpdir(), "chronofish-web-test-"));
  await esbuild.build({
    entryPoints: [
      path.join(root, "src/ai-layout.ts"),
      path.join(root, "src/ai-snapshot.ts"),
      path.join(root, "src/engine-io.ts")
    ],
    outdir,
    bundle: false,
    format: "esm",
    platform: "node",
    target: "es2022",
    logLevel: "silent"
  });
  return {
    aiLayout: pathToFileURL(path.join(outdir, "ai-layout.js")).href,
    aiSnapshot: pathToFileURL(path.join(outdir, "ai-snapshot.js")).href
  };
}
