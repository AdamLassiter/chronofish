import { readWasmString, writeWasmString } from "./engine-io.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import type { ChronofishEngine } from "./types.js";
import type { TrainingSample } from "./training-gpu.js";

/** Owns the Rust/WASM contract used by the browser training coordinator. */
export class GpuTrainingBinding {
  private enginePromise: Promise<ChronofishEngine> | null = null;
  private loaded: ChronofishEngine | null = null;

  async engine(): Promise<ChronofishEngine> {
    this.enginePromise ??= instantiateChronofishWasm("./chronofish_engine.wasm")
      .then((instance) => {
        this.loaded = instance.exports as unknown as ChronofishEngine;
        return this.loaded;
      });
    return this.enginePromise;
  }

  loadedEngine(): ChronofishEngine {
    if (!this.loaded) {
      throw new Error("Training engine is not initialized.");
    }
    return this.loaded;
  }

  async normalizeConfig<T>(config: unknown): Promise<T> {
    return this.jsonInput<T>(await this.engine(), "chronofish_normalize_training_config_json", config);
  }

  metricsSummary<T>(metrics: unknown): T {
    return this.jsonInput<T>(this.loadedEngine(), "chronofish_training_metrics_summary_json", metrics);
  }

  async dedupeSamples(samples: TrainingSample[]): Promise<TrainingSample[]> {
    return this.jsonInput<TrainingSample[]>(
      await this.engine(),
      "chronofish_dedupe_training_samples_json",
      samplesForEngine(samples)
    );
  }

  async trainingSamples<T>(games: unknown): Promise<T> {
    return this.jsonInput<T>(await this.engine(), "chronofish_training_samples_json", games);
  }

  async appendReplaySamples(
    buffer: TrainingSample[],
    samples: TrainingSample[],
    maxBuffer: number
  ): Promise<TrainingSample[]> {
    return this.jsonInput<TrainingSample[]>(
      await this.engine(),
      "chronofish_append_replay_samples_json",
      { buffer: samplesForEngine(buffer), samples: samplesForEngine(samples) },
      maxBuffer
    );
  }

  async labelSourceCounts(samples: TrainingSample[]): Promise<Record<string, number>> {
    return this.jsonInput<Record<string, number>>(
      await this.engine(),
      "chronofish_label_source_counts_json",
      samplesForEngine(samples)
    );
  }

  trainingLabelPolicy<T>(): T {
    return this.jsonOutput<T>(this.loadedEngine(), "chronofish_training_label_policy_json");
  }

  workerRequestTimeout(payload: unknown): number {
    return this.jsonNumber("chronofish_training_worker_request_timeout_ms_json", payload);
  }

  workerSearchTime(payload: unknown): number {
    return this.jsonNumber("chronofish_training_worker_search_time_ms_json", payload);
  }

  samplePlies(index: number, encodeOnly: boolean): number {
    return this.callNumber("chronofish_training_sample_plies", index, encodeOnly ? 1 : 0);
  }

  sampleSeed(prefix: string, index: number, salt: number): number {
    return this.textNumber("chronofish_training_sample_seed", prefix, index, salt) >>> 0;
  }

  searchSeed(value: unknown, salt: number): number {
    return this.jsonNumber("chronofish_training_search_seed_json", value, salt) >>> 0;
  }

  labelWorkerCount(jobCount: number, requestedWorkers: number, hardwareConcurrency: number): number {
    return this.callNumber("chronofish_training_label_worker_count", jobCount, requestedWorkers, hardwareConcurrency);
  }

  lossLogValidationUpdate<T>(validation: unknown, event: string, example: unknown): T {
    return this.jsonInput<T>(
      this.loadedEngine(),
      "chronofish_loss_log_validation_update_json",
      { validation, event, example }
    );
  }

  lossLogReplayLogs<T>(logs: unknown, limit: number): T {
    return this.jsonInput<T>(this.loadedEngine(), "chronofish_loss_log_replay_logs_json", { logs, limit });
  }

  movePlanKey(moves: unknown): string {
    return this.jsonString("chronofish_gpu_move_plan_key_json", moves ?? []);
  }

  splitWork(total: number, workers: number): number[] {
    return this.jsonOutput<number[]>(this.loadedEngine(), "chronofish_training_split_work_json", total, workers);
  }

  takeSampleBatches<T>(batches: unknown, target: number): T {
    return this.jsonInput<T>(this.loadedEngine(), "chronofish_take_training_sample_batches_json", batches, target);
  }

  compactSamples<T>(samples: unknown): T {
    return this.jsonInput<T>(this.loadedEngine(), "chronofish_compact_training_samples_json", samples);
  }

  gpuTrainingWorkerCount(total: number, requestedWorkers: number): number {
    return this.callNumber("chronofish_gpu_training_worker_count", total, requestedWorkers);
  }

  gpuDuelTrainingWorkerCount(total: number, searchWorkers: number, selfPlayWorkers: number): number {
    return this.callNumber("chronofish_gpu_duel_training_worker_count", total, searchWorkers, selfPlayWorkers);
  }

  gpuWarmupPlies(workerIndex: number): number {
    return this.callNumber("chronofish_gpu_warmup_plies", workerIndex);
  }

  gpuRolloutMaxPlies(target: number, workerIndex: number): number {
    return this.callNumber("chronofish_gpu_rollout_max_plies", target, workerIndex);
  }

  rolloutPlyOffset(ply: number, workerIndex: number): number {
    return this.callNumber("chronofish_gpu_rollout_ply_offset", ply, workerIndex);
  }

  gpuWarmupSearchConfig<T>(depth: number, nodes: number, timeMs: number, temperature: number): T {
    return this.jsonOutput<T>(
      this.loadedEngine(),
      "chronofish_gpu_warmup_search_config_json",
      depth,
      nodes,
      timeMs,
      temperature
    );
  }

  gpuPositionGenerationSearchConfig<T>(depth: number, nodes: number, temperature: number): T {
    return this.jsonOutput<T>(
      this.loadedEngine(),
      "chronofish_gpu_position_generation_search_config_json",
      depth,
      nodes,
      temperature
    );
  }

  curriculumSearchConfig<T>(depth: number, nodes: number, temperature: number, index: number): T {
    return this.jsonOutput<T>(
      this.loadedEngine(),
      "chronofish_curriculum_search_config_json",
      depth,
      nodes,
      temperature,
      index
    );
  }

  tacticalSearchConfig<T>(depth: number, nodes: number, temperature: number, attempt: number): T {
    return this.jsonOutput<T>(
      this.loadedEngine(),
      "chronofish_tactical_search_config_json",
      depth,
      nodes,
      temperature,
      attempt
    );
  }

  jsonValue<T>(operation: string, input: unknown, ...args: number[]): T {
    return this.jsonInput<T>(this.loadedEngine(), operation, input, ...args);
  }

  async asyncJsonValue<T>(operation: string, input: unknown, ...args: number[]): Promise<T> {
    return this.jsonInput<T>(await this.engine(), operation, input, ...args);
  }

  numericValue(operation: string, ...args: number[]): number {
    return this.callNumber(operation, ...args);
  }

  resultValue<T>(operation: string, ...args: number[]): T {
    return this.jsonOutput<T>(this.loadedEngine(), operation, ...args);
  }

  jsonNumericValue(operation: string, input: unknown, ...args: number[]): number {
    return this.jsonNumber(operation, input, ...args);
  }

  jsonBooleanValue(operation: string, input: unknown, ...args: number[]): boolean {
    return this.jsonNumericValue(operation, input, ...args) !== 0;
  }

  jsonTextValue(operation: string, input: unknown, ...args: number[]): string {
    const engine = this.loadedEngine();
    const { ptr, len } = writeWasmString(engine, JSON.stringify(input));
    try {
      const output = this.call(engine, operation, ptr, len, ...args);
      if (!output) {
        throw new Error(readWasmString(engine, engine.chronofish_last_message()));
      }
      return readWasmString(engine, output);
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
  }

  textResultValue<T>(operation: string, input: string, ...args: number[]): T {
    const engine = this.loadedEngine();
    const { ptr, len } = writeWasmString(engine, input);
    try {
      const output = this.call(engine, operation, ptr, len, ...args);
      if (!output) {
        throw new Error(readWasmString(engine, engine.chronofish_last_message()));
      }
      return JSON.parse(readWasmString(engine, output)) as T;
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
  }

  private jsonInput<T>(
    engine: ChronofishEngine,
    operation: string,
    input: unknown,
    ...args: number[]
  ): T {
    const { ptr, len } = writeWasmString(engine, JSON.stringify(input));
    try {
      const output = (engine as unknown as Record<string, (...values: number[]) => number>)[operation]?.(ptr, len, ...args);
      if (!output) {
        throw new Error(readWasmString(engine, engine.chronofish_last_message()));
      }
      return JSON.parse(readWasmString(engine, output)) as T;
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
  }

  private jsonOutput<T>(engine: ChronofishEngine, operation: string, ...args: number[]): T {
    const output = this.call(engine, operation, ...args);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as T;
  }

  private jsonNumber(operation: string, input: unknown, ...args: number[]): number {
    const engine = this.loadedEngine();
    const { ptr, len } = writeWasmString(engine, JSON.stringify(input));
    try {
      const value = this.call(engine, operation, ptr, len, ...args);
      if (!value) {
        throw new Error(readWasmString(engine, engine.chronofish_last_message()));
      }
      return value;
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
  }

  private textNumber(operation: string, input: string, ...args: number[]): number {
    const engine = this.loadedEngine();
    const { ptr, len } = writeWasmString(engine, input);
    try {
      return this.call(engine, operation, ptr, len, ...args);
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
  }

  private callNumber(operation: string, ...args: number[]): number {
    return this.call(this.loadedEngine(), operation, ...args);
  }

  private jsonString(operation: string, input: unknown): string {
    const engine = this.loadedEngine();
    const { ptr, len } = writeWasmString(engine, JSON.stringify(input));
    try {
      const output = this.call(engine, operation, ptr, len);
      if (!output) {
        throw new Error(readWasmString(engine, engine.chronofish_last_message()));
      }
      return readWasmString(engine, output);
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
  }

  private call(engine: ChronofishEngine, operation: string, ...args: number[]): number {
    const callback = (engine as unknown as Record<string, (...values: number[]) => number>)[operation];
    if (!callback) {
      throw new Error(`Training engine does not export ${operation}.`);
    }
    return callback(...args);
  }
}

function samplesForEngine(samples: TrainingSample[]): TrainingSample[] {
  return samples.map((sample) => ({
    ...sample,
    features: Array.from(sample.features ?? [])
  }));
}
