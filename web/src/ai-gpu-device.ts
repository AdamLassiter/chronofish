import { GPUBufferUsage } from "./ai-worker-types.js";

let pipelineCaches = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();

export function storageBuffer(device: GPUDevice, data: ArrayBuffer | ArrayBufferView, usage: number): GPUBuffer {
  const byteLength = data.byteLength;
  const buffer = device.createBuffer({
    size: align4(byteLength),
    usage: usage | GPUBufferUsage.COPY_DST
  });
  if (data instanceof ArrayBuffer) {
    device.queue.writeBuffer(buffer, 0, data);
  } else {
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  }
  return buffer;
}

export async function requestHighLimitDevice(adapter: GPUAdapter): Promise<GPUDevice> {
  const requiredLimits: Record<string, number> = {};
  const requiredFeatures: GPUFeatureName[] = [];
  for (const key of ["maxStorageBufferBindingSize", "maxBufferSize"] as const) {
    const value = adapter.limits[key];
    if (Number.isFinite(value) && value > 0) {
      requiredLimits[key] = value;
    }
  }
  if (adapter.features?.has("timestamp-query" as GPUFeatureName)) {
    requiredFeatures.push("timestamp-query" as GPUFeatureName);
  }
  if (adapter.features?.has("shader-f16" as GPUFeatureName)) {
    requiredFeatures.push("shader-f16" as GPUFeatureName);
  }
  if (Object.keys(requiredLimits).length === 0 && requiredFeatures.length === 0) {
    return adapter.requestDevice();
  }
  try {
    return await adapter.requestDevice({ requiredLimits, requiredFeatures });
  } catch {
    return adapter.requestDevice();
  }
}

export async function createComputePipelineChecked(device: GPUDevice, label: string, code: string, entryPoint: string): Promise<GPUComputePipeline> {
  const cacheKey = `${label}:${entryPoint}`;
  let pipelineCache = pipelineCaches.get(device);
  if (!pipelineCache) {
    pipelineCache = new Map<string, GPUComputePipeline>();
    pipelineCaches.set(device, pipelineCache);
  }
  const cached = pipelineCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const module = device.createShaderModule({ label: `${label}.module`, code });
  if (module.getCompilationInfo) {
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message: GPUCompilationMessage) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(formatShaderErrors(label, errors));
    }
  }
  const pipeline = device.createComputePipeline({
    label,
    layout: "auto",
    compute: { module, entryPoint }
  });
  pipelineCache.set(cacheKey, pipeline);
  return pipeline;
}

export function clearComputePipelineCache(): void {
  pipelineCaches = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();
}

export function formatShaderErrors(label: string, errors: GPUCompilationMessage[]): string {
  return `${label} shader compilation failed: ${errors.map((error) =>
    `line ${error.lineNum ?? "?"}, column ${error.linePos ?? "?"}: ${error.message}`
  ).join("; ")}`;
}

export function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}
