import { writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PROJECTION_SIZE = 2048;
const PROJECTION_SEED = 2166136261;
const HIDDEN_LAYERS = [1024, 512, 256];
const OUTPUT_SIZE = HIDDEN_LAYERS.at(-1) + 1;

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const outputPath = path.join(root, "engine/models/gpu-v1/value-model.cfnn");
const hiddenWeights = initialHiddenWeights(PROJECTION_SIZE, HIDDEN_LAYERS);
const outputWeights = new Float32Array(OUTPUT_SIZE);
const byteLength = 40
  + HIDDEN_LAYERS.length * Uint32Array.BYTES_PER_ELEMENT
  + (hiddenWeights.length + outputWeights.length) * Float32Array.BYTES_PER_ELEMENT;
const buffer = new ArrayBuffer(byteLength);
const view = new DataView(buffer);
let cursor = 0;

for (const byte of Buffer.from("CFNN")) {
  view.setUint8(cursor, byte);
  cursor += 1;
}
cursor = writeU32(view, cursor, 4);
cursor = writeU32(view, cursor, PROJECTION_SIZE);
cursor = writeU32(view, cursor, PROJECTION_SEED);
cursor = writeU32(view, cursor, HIDDEN_LAYERS.length);
cursor = writeU32(view, cursor, outputWeights.length);
cursor = writeU32(view, cursor, 0);
cursor = writeF32(view, cursor, 1);
cursor = writeF32(view, cursor, 0);
for (const layer of HIDDEN_LAYERS) {
  cursor = writeU32(view, cursor, layer);
}
cursor = writeU32(view, cursor, hiddenWeights.length);
for (const value of hiddenWeights) {
  cursor = writeF32(view, cursor, value);
}
for (const value of outputWeights) {
  cursor = writeF32(view, cursor, value);
}
if (cursor !== buffer.byteLength) {
  throw new Error(`CFNN size mismatch: wrote ${cursor} bytes into ${buffer.byteLength}.`);
}
await writeFile(outputPath, new Uint8Array(buffer));
console.log(`Initialized ${path.relative(root, outputPath)} (${buffer.byteLength} bytes).`);

function initialHiddenWeights(inputSize, hiddenLayers) {
  const weights = [];
  let previousSize = inputSize;
  for (let layerIndex = 0; layerIndex < hiddenLayers.length; layerIndex += 1) {
    const layerSize = hiddenLayers[layerIndex];
    const scale = Math.sqrt(2 / previousSize);
    for (let output = 0; output < layerSize; output += 1) {
      for (let input = 0; input < previousSize; input += 1) {
        const hash = projectionHash(input, output + layerIndex * 4099, PROJECTION_SEED);
        weights.push((((hash / 0xffffffff) * 2) - 1) * scale);
      }
      weights.push(0);
    }
    previousSize = layerSize;
  }
  return new Float32Array(weights);
}

function projectionHash(rawIndex, projectionIndex, seed) {
  let hash = (seed ^ rawIndex) >>> 0;
  hash = Math.imul(hash, 16777619) >>> 0;
  hash = (hash ^ projectionIndex) >>> 0;
  hash = Math.imul(hash, 16777619) >>> 0;
  return (hash ^ (hash >>> 16)) >>> 0;
}

function writeU32(view, offset, value) {
  view.setUint32(offset, value, true);
  return offset + 4;
}

function writeF32(view, offset, value) {
  view.setFloat32(offset, value, true);
  return offset + 4;
}
