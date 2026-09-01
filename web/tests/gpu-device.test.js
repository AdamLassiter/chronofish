import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import * as esbuild from "esbuild";

const root = path.resolve(import.meta.dirname, "..");
const modulePath = await buildTestModule();
const {
  clearComputePipelineCache,
  createComputePipelineChecked,
  storageBuffer
} = await import(modulePath);

test("GPU storage uploads typed-array windows without copying them first", () => {
  const source = new ArrayBuffer(16);
  const view = new Uint8Array(source, 4, 6);
  let upload = null;
  const buffer = { label: "storage" };
  const device = {
    createBuffer(descriptor) {
      assert.equal(descriptor.size, 8);
      return buffer;
    },
    queue: {
      writeBuffer(...arguments_) {
        upload = arguments_;
      }
    }
  };

  assert.equal(storageBuffer(device, view, 128), buffer);
  assert.equal(upload[2], source);
  assert.equal(upload[3], 4);
  assert.equal(upload[4], 6);
});

test("GPU compute pipeline caches are isolated by device", async () => {
  clearComputePipelineCache();
  const first = fakeDevice("first");
  const second = fakeDevice("second");

  const firstPipeline = await createComputePipelineChecked(first.device, "movegen", "shader", "main");
  assert.equal(await createComputePipelineChecked(first.device, "movegen", "shader", "main"), firstPipeline);
  const secondPipeline = await createComputePipelineChecked(second.device, "movegen", "shader", "main");

  assert.notEqual(secondPipeline, firstPipeline);
  assert.equal(first.pipelineCreations(), 1);
  assert.equal(second.pipelineCreations(), 1);

  clearComputePipelineCache();
  assert.notEqual(
    await createComputePipelineChecked(first.device, "movegen", "shader", "main"),
    firstPipeline
  );
  assert.equal(first.pipelineCreations(), 2);
});

function fakeDevice(name) {
  let pipelines = 0;
  return {
    device: {
      createShaderModule() {
        return {
          async getCompilationInfo() {
            return { messages: [] };
          }
        };
      },
      createComputePipeline() {
        pipelines += 1;
        return { name, pipelines };
      }
    },
    pipelineCreations() {
      return pipelines;
    }
  };
}

async function buildTestModule() {
  const outdir = await mkdtemp(path.join(os.tmpdir(), "chronofish-gpu-device-test-"));
  await esbuild.build({
    entryPoints: [path.join(root, "src/ai-gpu-device.ts")],
    outdir,
    bundle: true,
    format: "esm",
    platform: "node",
    target: "es2022",
    logLevel: "silent"
  });
  return pathToFileURL(path.join(outdir, "ai-gpu-device.js")).href;
}
