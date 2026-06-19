import { capitalize } from "./board.js";
import { elements } from "./dom.js";
import type { ChronofishEngine, GameSnapshot } from "./types.js";

type TrainingTone = "neutral" | "ready" | "warn" | "error" | "active";
type LabelKind = "search" | "outcome" | "distilled" | string;
type LabelPhase = "positions" | "labels" | string;

interface TrainingControllerOptions {
  getEngine(): ChronofishEngine | null;
  getGame(): GameSnapshot;
  resetAiWorker(): void;
}

export interface TrainingController {
  loadStatus(): Promise<void>;
  renderButtons(): void;
  openModal(): void;
  closeModal(): void;
  start(): Promise<void>;
  stop(): void;
}

interface TrainingGpuProfile {
  name: string;
  limits: GPUSupportedLimits;
  config: TrainingGpuProfileConfig;
}

interface TrainingGpuProfileConfig {
  maxSamples: number;
  samples: number;
  maxSelfPlayWorkers: number;
  selfPlayWorkers: number;
  maxSearchWorkers: number;
  searchWorkers: number;
  maxNodes: number;
  nodes: number;
  maxEpochs: number;
  epochs: number;
  maxBuffer: number;
  buffer: number;
  maxBatch: number;
  batch: number;
  maxValidationInterval: number;
  validationInterval: number;
}

interface TrainingConfig {
  trainingSubject: string;
  trainingModes: string[];
  samples: number;
  selfPlayWorkers: number;
  searchWorkers: number;
  explorationTemperature: number;
  depth: number;
  nodes: number;
  learningRate: number;
  epochs: number;
  maxBuffer: number;
  batchSize: number;
  validationSplit: number;
  lossLogReplay: number;
  validationInterval: number;
  patience: number;
  weightDecay: number;
  labelWorkers: number;
  cpuDepth: number;
  cpuNodes: number;
  cpuTrainingTimeMs: number;
  cpuCandidates: number;
  cpuFinalists: number;
  cpuPairBatch: number;
  cpuOpponentVariants: number;
  cpuScreeningOpponentVariants: number;
  cpuRoundsPerVariant: number;
  cpuHallOfFameEntries: number;
  cpuLeagueContenders: number;
  cpuLeagueHallOfFameEntries: number;
  cpuMinPairs: number;
  cpuMaxPairs: number;
  cpuDrawWindow: number;
  cpuDrawRateLimit: number;
  cpuMaxMatchPlies: number;
  cpuMaxMatchTimeMs: number;
  cpuMaxGenerationsWithoutCandidate: number;
  cpuWorkers: number;
  cpuTrainSeconds: number;
}

interface CpuTrainingParameters {
  timeMs?: number;
  nodes?: number;
  candidates?: number | null;
  finalists?: number | null;
  pairBatch?: number | null;
  opponentVariants?: number;
  screeningOpponentVariants?: number;
  roundsPerVariant?: number;
  hallOfFameEntries?: number;
  leagueContenders?: number;
  leagueHallOfFameEntries?: number;
  minPairs?: number;
  maxPairs?: number;
  drawWindow?: number;
  drawRateLimit?: number;
  maxMatchPlies?: number;
  maxMatchTimeMs?: number;
  maxGenerationsWithoutCandidate?: number;
}

interface TrainingMetric {
  label: string;
  value: string;
  title?: string | null;
}

interface TrainingStatusOptions {
  title: string;
  tone?: TrainingTone;
  metrics?: TrainingMetric[];
}

interface TrainingProgressEntry {
  labelKind: LabelKind;
  labelPhase: LabelPhase;
  collected: number;
  sampleCount: number;
  labelWorkers: number;
}

interface TrainingProgressUpdate {
  labelKind?: LabelKind | undefined;
  labelPhase?: LabelPhase | undefined;
  collected?: number | undefined;
  sampleCount?: number | undefined;
  labelWorkers?: number | undefined;
}

interface TrainingStatusPayload {
  enabled?: boolean;
  modelPath?: string;
  modelPresent?: boolean;
  modelBytes?: number;
  modelHash?: string;
  resolvedModelPath?: string;
  cpuTrainingPresent?: boolean;
  cpuTrainingHash?: string;
  resolvedCpuTrainingPath?: string;
}

const TRAINING_STATUS_TIMEOUT_MS = 1500;

interface TrainingReplacementPayload {
  changed?: boolean;
  resolvedModelPath?: string;
  modelPath?: string;
  newHash?: string;
  error?: string;
}

interface LossLogValidation {
  checked: number;
  changed: number;
  skipped?: number;
  failed?: boolean;
}

interface TrainingRunMetrics {
  totalMs?: number;
  phases?: Record<string, number>;
  sampleRates?: Record<string, number>;
  lossLogValidation?: LossLogValidation | null;
}

interface TrainingWorkerMessage {
  id: number;
  ok?: boolean;
  type?: string;
  error?: string;
  model?: ArrayBuffer;
  cpuParameters?: string;
  cpuScore?: number;
  loss?: number;
  epoch?: number;
  collected?: number;
  sampleCount?: number;
  labelWorkers?: number;
  labelKind?: LabelKind;
  labelPhase?: LabelPhase;
  selfPlayWorkers?: number;
  searchWorkers?: number;
  bufferSize?: number;
  labelCounts?: Record<string, number>;
  batchSize?: number;
  batchesPerSubmit?: number;
  validationInterval?: number;
  replaySize?: number;
  validationLoss?: number;
  bestValidationLoss?: number;
  epochsWithoutImprovement?: number;
  earlyStopReason?: string;
  nonZeroWeights?: number;
  gpuPhase?: string;
  metrics?: TrainingRunMetrics;
  lossLogValidation?: LossLogValidation;
  validation?: LossLogValidation;
}

interface TrainingWorkerRequest {
  id: number;
  type: "train" | "validateLossLogs";
  game?: GameSnapshot;
  config: TrainingConfig;
}

export function createTrainingController({ getEngine, getGame, resetAiWorker }: TrainingControllerOptions): TrainingController {
  const openTrainingButton = requireElement(elements.openTrainingButton, "open-training");
  const trainingModal = requireElement(elements.trainingModal, "training-modal");
  const trainingPanel = requireElement(elements.trainingPanel, "training-panel");
  const startTrainingButton = requireElement(elements.startTrainingButton, "start-training");
  const stopTrainingButton = requireElement(elements.stopTrainingButton, "stop-training");
  const trainingStatus = requireElement(elements.trainingStatus, "training-status");
  const trainingStatusView = requireElement(elements.trainingStatusView, "training-status-view");
  const trainingProgress = requireElement(elements.trainingProgress, "training-progress");
  const trainingModeSelect = requireElement(elements.trainingModeSelect, "training-mode");
  const samplesInput = requireElement(elements.trainingSamplesInput, "training-samples");
  const selfPlayWorkersInput = requireElement(elements.trainingSelfPlayWorkersInput, "training-self-play-workers");
  const searchWorkersInput = requireElement(elements.trainingSearchWorkersInput, "training-search-workers");
  const temperatureInput = requireElement(elements.trainingTemperatureInput, "training-temperature");
  const depthInput = requireElement(elements.trainingDepthInput, "training-depth");
  const nodesInput = requireElement(elements.trainingNodesInput, "training-nodes");
  const rateInput = requireElement(elements.trainingRateInput, "training-rate");
  const epochsInput = requireElement(elements.trainingEpochsInput, "training-epochs");
  const bufferInput = requireElement(elements.trainingBufferInput, "training-buffer");
  const batchInput = requireElement(elements.trainingBatchInput, "training-batch");
  const validationInput = requireElement(elements.trainingValidationInput, "training-validation");
  const lossLogReplayInput = requireElement(elements.trainingLossLogReplayInput, "training-loss-log-replay");
  const validationIntervalInput = requireElement(elements.trainingValidationIntervalInput, "training-validation-interval");
  const patienceInput = requireElement(elements.trainingPatienceInput, "training-patience");
  const decayInput = requireElement(elements.trainingDecayInput, "training-decay");
  const cpuModeSelect = requireElement(elements.trainingCpuModeSelect, "training-cpu-mode");
  const cpuDepthInput = requireElement(elements.trainingCpuDepthInput, "training-cpu-depth");
  const cpuNodesInput = requireElement(elements.trainingCpuNodesInput, "training-cpu-nodes");
  const cpuTimeMsInput = requireElement(elements.trainingCpuTimeMsInput, "training-cpu-time-ms");
  const cpuCandidatesInput = requireElement(elements.trainingCpuCandidatesInput, "training-cpu-candidates");
  const cpuFinalistsInput = requireElement(elements.trainingCpuFinalistsInput, "training-cpu-finalists");
  const cpuPairBatchInput = requireElement(elements.trainingCpuPairBatchInput, "training-cpu-pair-batch");
  const cpuOpponentVariantsInput = requireElement(elements.trainingCpuOpponentVariantsInput, "training-cpu-opponent-variants");
  const cpuScreeningOpponentVariantsInput = requireElement(elements.trainingCpuScreeningOpponentVariantsInput, "training-cpu-screening-opponent-variants");
  const cpuRoundsPerVariantInput = requireElement(elements.trainingCpuRoundsPerVariantInput, "training-cpu-rounds-per-variant");
  const cpuHallOfFameEntriesInput = requireElement(elements.trainingCpuHallOfFameEntriesInput, "training-cpu-hall-of-fame-entries");
  const cpuLeagueContendersInput = requireElement(elements.trainingCpuLeagueContendersInput, "training-cpu-league-contenders");
  const cpuLeagueHallOfFameEntriesInput = requireElement(elements.trainingCpuLeagueHallOfFameEntriesInput, "training-cpu-league-hall-of-fame-entries");
  const cpuMinPairsInput = requireElement(elements.trainingCpuMinPairsInput, "training-cpu-min-pairs");
  const cpuMaxPairsInput = requireElement(elements.trainingCpuMaxPairsInput, "training-cpu-max-pairs");
  const cpuDrawWindowInput = requireElement(elements.trainingCpuDrawWindowInput, "training-cpu-draw-window");
  const cpuDrawRateLimitInput = requireElement(elements.trainingCpuDrawRateLimitInput, "training-cpu-draw-rate-limit");
  const cpuMaxMatchPliesInput = requireElement(elements.trainingCpuMaxMatchPliesInput, "training-cpu-max-match-plies");
  const cpuMaxMatchTimeMsInput = requireElement(elements.trainingCpuMaxMatchTimeMsInput, "training-cpu-max-match-time-ms");
  const cpuMaxGenerationsWithoutCandidateInput = requireElement(elements.trainingCpuMaxGenerationsWithoutCandidateInput, "training-cpu-max-generations-without-candidate");
  const cpuWorkersInput = requireElement(elements.trainingCpuWorkersInput, "training-cpu-workers");
  const cpuSecondsInput = requireElement(elements.trainingCpuSecondsInput, "training-cpu-seconds");

  let trainingWorker: Worker | null = null;
  let trainingRequestId = 0;
  let trainingEnabled = false;
  let trainingRunning = false;
  let trainingCycle = 0;
  let trainingGpuProfile: TrainingGpuProfile | null = null;
  let trainingGpuProfileApplied = false;
  let cpuTrainingParametersApplied = false;
  let selectedTrainingTab = "gpu";
  const trainingProgressState = new Map<LabelKind, TrainingProgressEntry>();

  elements.trainingTabButtons.forEach((button) => {
    button.addEventListener("click", () => selectTrainingTab(button.dataset.trainingTab ?? "gpu"));
  });

  function ensureTrainingWorker(): Worker {
    if (!trainingWorker) {
      trainingWorker = new Worker("./training-worker.js", { type: "module" });
      trainingWorker.addEventListener("message", handleTrainingWorkerMessage);
    }
    return trainingWorker;
  }

  async function loadTrainingStatus(): Promise<void> {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), TRAINING_STATUS_TIMEOUT_MS);
    try {
      const response = await fetch("/api/training/status", { signal: controller.signal });
      if (!response.ok) {
        throw new Error("Training endpoints disabled");
      }
      const payload = await response.json() as TrainingStatusPayload;
      if (payload.enabled !== true || payload.modelPath !== "engine/models/gpu-v1/value-model.cfnn") {
        throw new Error("Training endpoints disabled");
      }
      trainingEnabled = true;
      openTrainingButton.hidden = false;
      trainingPanel.hidden = false;
      await applyTrainingGpuProfile();
      await applyCpuTrainingParameters();
      setTrainingStatus({
        title: payload.modelPresent ? "Model Ready" : "No Model",
        tone: payload.modelPresent ? "ready" : "warn",
        metrics: filterMetrics([
          trainingMetric("Bytes", payload.modelPresent ? payload.modelBytes ?? 0 : null),
          trainingMetric("Hash", compactHash(payload.modelHash), payload.modelHash),
          trainingMetric("GPU", trainingGpuProfile?.name),
          trainingMetric("CPU Training", compactHash(payload.cpuTrainingHash), payload.resolvedCpuTrainingPath),
          trainingMetric("Path", compactPath(payload.resolvedModelPath), payload.resolvedModelPath)
        ])
      });
      renderTrainingButtons();
    } catch {
      trainingEnabled = false;
      openTrainingButton.hidden = true;
      trainingModal.hidden = true;
      trainingPanel.hidden = true;
      resetTrainingProgress();
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function trainingConfig(): TrainingConfig {
    const caps = trainingGpuProfile?.config;
    const cpuWorkerMax = Math.max(1, Math.min(navigator.hardwareConcurrency ?? 4, 32));
    return {
      trainingSubject: selectedTrainingTab === "cpu" ? "cpu" : "gpu",
      // GPU
      trainingModes: selectedTrainingModes(selectedTrainingTab === "cpu" ? cpuModeSelect : trainingModeSelect, selectedTrainingTab === "cpu" ? ["vsCpu"] : ["vsGpu", "self"]),
      samples: clampNumber(samplesInput.value, 1, caps?.maxSamples ?? 512, caps?.samples ?? 64),
      selfPlayWorkers: clampNumber(selfPlayWorkersInput.value, 1, caps?.maxSelfPlayWorkers ?? 8, caps?.selfPlayWorkers ?? 2),
      searchWorkers: clampNumber(searchWorkersInput.value, 1, caps?.maxSearchWorkers ?? 16, caps?.searchWorkers ?? 2),
      explorationTemperature: clampNumber(temperatureInput.value, 0, 2, 0.25),
      depth: clampNumber(depthInput.value, 1, 8, 5),
      nodes: clampNumber(nodesInput.value, 1, caps?.maxNodes ?? 65536, caps?.nodes ?? 16384),
      learningRate: clampNumber(rateInput.value, 0.0001, 0.1, 0.01),
      epochs: clampNumber(epochsInput.value, 1, caps?.maxEpochs ?? 65536, caps?.epochs ?? 8192),
      maxBuffer: clampNumber(bufferInput.value, 16, caps?.maxBuffer ?? 8192, caps?.buffer ?? 4096),
      batchSize: clampNumber(batchInput.value, 16, caps?.maxBatch ?? 4096, caps?.batch ?? 1024),
      validationSplit: clampNumber(validationInput.value, 0, 0.3, 0.1),
      lossLogReplay: clampNumber(lossLogReplayInput.value, 0, 32, 4),
      validationInterval: clampNumber(validationIntervalInput.value, 16, caps?.maxValidationInterval ?? 4096, caps?.validationInterval ?? 256),
      patience: clampNumber(patienceInput.value, 1, 64, 12),
      weightDecay: clampNumber(decayInput.value, 0, 0.01, 0.00001),
      labelWorkers: autoTrainingWorkers(),
      // CPU
      cpuDepth: clampNumber(cpuDepthInput.value, 1, 16, 4),
      cpuNodes: clampNumber(cpuNodesInput.value, 1, 131072, 16384),
      cpuTrainingTimeMs: clampNumber(cpuTimeMsInput.value, 1, 600000, 10000),
      cpuCandidates: clampNumber(cpuCandidatesInput.value, 1, 256, 8),
      cpuFinalists: clampNumber(cpuFinalistsInput.value, 1, 64, 1),
      cpuPairBatch: clampNumber(cpuPairBatchInput.value, 1, 64, 4),
      cpuOpponentVariants: clampNumber(cpuOpponentVariantsInput.value, 1, 128, 8),
      cpuScreeningOpponentVariants: clampNumber(cpuScreeningOpponentVariantsInput.value, 1, 128, 2),
      cpuRoundsPerVariant: clampNumber(cpuRoundsPerVariantInput.value, 1, 64, 1),
      cpuHallOfFameEntries: clampNumber(cpuHallOfFameEntriesInput.value, 0, 64, 1),
      cpuLeagueContenders: clampNumber(cpuLeagueContendersInput.value, 1, 64, 2),
      cpuLeagueHallOfFameEntries: clampNumber(cpuLeagueHallOfFameEntriesInput.value, 0, 64, 2),
      cpuMinPairs: clampNumber(cpuMinPairsInput.value, 1, 256, 2),
      cpuMaxPairs: clampNumber(cpuMaxPairsInput.value, 1, 512, 8),
      cpuDrawWindow: clampNumber(cpuDrawWindowInput.value, 1, 128, 4),
      cpuDrawRateLimit: clampNumber(cpuDrawRateLimitInput.value, 0, 1, 0.8),
      cpuMaxMatchPlies: clampNumber(cpuMaxMatchPliesInput.value, 1, 512, 40),
      cpuMaxMatchTimeMs: clampNumber(cpuMaxMatchTimeMsInput.value, 0, 3600000, 0),
      cpuMaxGenerationsWithoutCandidate: clampNumber(cpuMaxGenerationsWithoutCandidateInput.value, 1, 256, 2),
      cpuWorkers: clampNumber(cpuWorkersInput.value, 1, cpuWorkerMax, Math.min(cpuWorkerMax, 16)),
      cpuTrainSeconds: clampNumber(cpuSecondsInput.value, 1, 86400, 3600)
    };
  }

  function selectTrainingTab(name: string): void {
    const selected = name === "cpu" ? "cpu" : "gpu";
    selectedTrainingTab = selected;
    for (const button of elements.trainingTabButtons) {
      button.setAttribute("aria-pressed", button.dataset.trainingTab === selected ? "true" : "false");
    }
    for (const panel of elements.trainingTabPanels) {
      panel.hidden = panel.dataset.trainingPanel !== selected;
    }
  }

  function selectedTrainingModes(select: HTMLSelectElement, fallback: string[]): string[] {
    const values = Array.from(select.selectedOptions).map((option) => option.value);
    return values.length > 0 ? values : fallback;
  }

  function autoTrainingWorkers(): number {
    const cores = navigator.hardwareConcurrency ?? 4;
    return Math.max(1, Math.min(cores - 1, 4));
  }

  async function applyTrainingGpuProfile(): Promise<void> {
    if (trainingGpuProfileApplied) {
      return;
    }
    trainingGpuProfileApplied = true;
    trainingGpuProfile = await detectTrainingGpuProfile();
    if (!trainingGpuProfile) {
      return;
    }
    const config = trainingGpuProfile.config;
    applyTrainingInputProfile(samplesInput, config.samples, config.maxSamples);
    applyTrainingInputProfile(selfPlayWorkersInput, config.selfPlayWorkers, config.maxSelfPlayWorkers);
    applyTrainingInputProfile(searchWorkersInput, config.searchWorkers, config.maxSearchWorkers);
    applyTrainingInputProfile(nodesInput, config.nodes, config.maxNodes);
    applyTrainingInputProfile(epochsInput, config.epochs, config.maxEpochs);
    applyTrainingInputProfile(bufferInput, config.buffer, config.maxBuffer);
    applyTrainingInputProfile(batchInput, config.batch, config.maxBatch);
    applyTrainingInputProfile(validationIntervalInput, config.validationInterval, config.maxValidationInterval);
  }

  async function detectTrainingGpuProfile(): Promise<TrainingGpuProfile | null> {
    if (!navigator.gpu?.requestAdapter) {
      return null;
    }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        return null;
      }
      const limits = adapter.limits;
      const maxStorageBinding = limits.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
      const maxBufferSize = limits.maxBufferSize ?? maxStorageBinding;
      const hardwareThreads = navigator.hardwareConcurrency ?? 4;
      const maxProjectedReplay = clampPowerOfTwo(Math.floor(maxStorageBinding / (2048 * Float32Array.BYTES_PER_ELEMENT)), 512, 16384);
      const maxBatchByActivation = clampPowerOfTwo(Math.floor(maxStorageBinding / (1024 * Float32Array.BYTES_PER_ELEMENT)), 512, 16384);
      const highMemory = maxStorageBinding >= 512 * 1024 * 1024 || maxBufferSize >= 1024 * 1024 * 1024;
      const mediumMemory = maxStorageBinding >= 256 * 1024 * 1024 || maxBufferSize >= 512 * 1024 * 1024;
      const maxWorkerBudget = Math.max(1, Math.min(highMemory ? 32 : 8, hardwareThreads - 1));
      const config: TrainingGpuProfileConfig = {
        maxSamples: highMemory ? 16384 : 8192,
        samples: highMemory ? 8192 : mediumMemory ? 4096 : 1024,
        maxSelfPlayWorkers: maxWorkerBudget,
        selfPlayWorkers: Math.max(1, Math.min(maxWorkerBudget, highMemory ? 16 : mediumMemory ? 8 : 4)),
        maxSearchWorkers: maxWorkerBudget,
        searchWorkers: Math.max(1, Math.min(maxWorkerBudget, highMemory ? 16 : mediumMemory ? 8 : 4)),
        maxNodes: highMemory ? 16384 : 8192,
        nodes: highMemory ? 8192 : mediumMemory ? 4096 : 1024,
        maxEpochs: 16384,
        epochs: highMemory ? 8192 : 4096,
        maxBuffer: Math.min(maxProjectedReplay, highMemory ? 16384 : 8192),
        buffer: Math.min(maxProjectedReplay, highMemory ? 8192 : mediumMemory ? 4096 : 1024),
        maxBatch: Math.min(maxBatchByActivation, highMemory ? 16384 : 8192),
        batch: Math.min(maxBatchByActivation, highMemory ? 8192 : mediumMemory ? 4096 : 1024),
        maxValidationInterval: 16384,
        validationInterval: highMemory ? 8192 : mediumMemory ? 4096 : 1024
      };
      const info = gpuAdapterName(adapter);
      console.info(`Detected training GPU profile: ${info}`, { limits, config });
      return {
        name: info,
        limits,
        config
      };
    } catch (error) {
      console.warn("GPU capability detection failed", error);
      return null;
    }
  }

  function applyTrainingInputProfile(input: HTMLInputElement, value: number, max: number): void {
    input.max = String(max);
    input.value = String(value);
  }

  async function applyCpuTrainingParameters(): Promise<void> {
    if (cpuTrainingParametersApplied) {
      return;
    }
    cpuTrainingParametersApplied = true;
    const parameters = await fetchCpuTrainingParameters();
    if (!parameters) {
      return;
    }
    applyCpuTrainingInput(cpuTimeMsInput, parameters.timeMs, 1, 600000, 10000);
    applyCpuTrainingInput(cpuNodesInput, parameters.nodes, 1, 131072, 8192);
    applyCpuTrainingInput(cpuCandidatesInput, parameters.candidates, 1, 256, 8);
    applyCpuTrainingInput(cpuFinalistsInput, parameters.finalists, 1, 64, 1);
    applyCpuTrainingInput(cpuPairBatchInput, parameters.pairBatch, 1, 64, 4);
    applyCpuTrainingInput(cpuOpponentVariantsInput, parameters.opponentVariants, 1, 128, 8);
    applyCpuTrainingInput(cpuScreeningOpponentVariantsInput, parameters.screeningOpponentVariants, 1, 128, 2);
    applyCpuTrainingInput(cpuRoundsPerVariantInput, parameters.roundsPerVariant, 1, 64, 1);
    applyCpuTrainingInput(cpuHallOfFameEntriesInput, parameters.hallOfFameEntries, 0, 64, 1);
    applyCpuTrainingInput(cpuLeagueContendersInput, parameters.leagueContenders, 1, 64, 2);
    applyCpuTrainingInput(cpuLeagueHallOfFameEntriesInput, parameters.leagueHallOfFameEntries, 0, 64, 2);
    applyCpuTrainingInput(cpuMinPairsInput, parameters.minPairs, 1, 256, 2);
    applyCpuTrainingInput(cpuMaxPairsInput, parameters.maxPairs, 1, 512, 8);
    applyCpuTrainingInput(cpuDrawWindowInput, parameters.drawWindow, 1, 128, 4);
    applyCpuTrainingInput(cpuDrawRateLimitInput, parameters.drawRateLimit, 0, 1, 0.8);
    applyCpuTrainingInput(cpuMaxMatchPliesInput, parameters.maxMatchPlies, 1, 512, 40);
    applyCpuTrainingInput(cpuMaxMatchTimeMsInput, parameters.maxMatchTimeMs, 0, 3600000, 0);
    applyCpuTrainingInput(cpuMaxGenerationsWithoutCandidateInput, parameters.maxGenerationsWithoutCandidate, 1, 256, 2);
  }

  async function fetchCpuTrainingParameters(): Promise<CpuTrainingParameters | null> {
    for (const path of ["/api/training/cpu-training", "/ai/training.json"]) {
      try {
        const response = await fetch(path, { cache: "no-store" });
        if (response.ok) {
          return await response.json() as CpuTrainingParameters;
        }
      } catch {
        // Try the next source.
      }
    }
    return null;
  }

  function applyCpuTrainingInput(input: HTMLInputElement, value: number | null | undefined, min: number, max: number, fallback: number): void {
    input.value = String(clampNumber(String(value ?? fallback), min, max, fallback));
  }

  function clampPowerOfTwo(value: number, min: number, max: number): number {
    const clamped = Math.max(min, Math.min(max, value || min));
    return 2 ** Math.floor(Math.log2(clamped));
  }

  function clampNumber(value: string, min: number, max: number, fallback: number): number {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return fallback;
    }
    return Math.min(max, Math.max(min, number));
  }

  function renderTrainingButtons(): void {
    if (!trainingEnabled) {
      return;
    }
    startTrainingButton.disabled = !getEngine() || !trainingEnabled || trainingRunning;
    stopTrainingButton.disabled = !trainingRunning;
  }

  function setTrainingStatus({ title, tone = "neutral", metrics = [] }: TrainingStatusOptions): void {
    const text = [title, ...metrics.map((metric) => `${metric.label}: ${metric.value}`)].join(". ");
    trainingStatus.textContent = text;
    trainingStatusView.dataset.tone = tone;
    const phase = document.createElement("div");
    phase.className = "training-status-phase";
    const dot = document.createElement("span");
    dot.className = "training-status-dot";
    const label = document.createElement("span");
    label.textContent = title;
    phase.append(dot, label);

    const metricGrid = document.createElement("div");
    metricGrid.className = "training-status-metrics";
    metricGrid.replaceChildren(...metrics.map((metric) => {
      const item = document.createElement("div");
      item.className = "training-status-metric";
      const metricLabel = document.createElement("span");
      metricLabel.className = "training-status-metric-label";
      metricLabel.textContent = metric.label;
      const metricValue = document.createElement("span");
      metricValue.className = "training-status-metric-value";
      metricValue.textContent = metric.value;
      if (metric.title) {
        metricValue.title = metric.title;
      }
      item.append(metricLabel, metricValue);
      return item;
    }));
    trainingStatusView.replaceChildren(phase, metricGrid);
  }

  function trainingMetric(label: string, value: unknown, title: string | null | undefined = null): TrainingMetric | null {
    if (value === null || value === undefined || value === "") {
      return null;
    }
    return { label, value: String(value), title };
  }

  function compactHash(hash: string | null | undefined): string | null {
    return hash ? String(hash).slice(0, 12) : null;
  }

  function compactPath(value: string | null | undefined): string | null {
    if (!value) {
      return null;
    }
    const parts = String(value).split("/");
    return parts.slice(-3).join("/");
  }

  function resetTrainingProgress(): void {
    trainingProgressState.clear();
    trainingProgress.hidden = true;
    trainingProgress.replaceChildren();
  }

  function updateTrainingProgress({ labelKind, labelPhase, collected, sampleCount, labelWorkers }: TrainingProgressUpdate): void {
    if (!labelKind || !Number.isFinite(sampleCount) || (sampleCount ?? 0) <= 0) {
      return;
    }
    const existing = trainingProgressState.get(labelKind);
    const total = sampleCount ?? existing?.sampleCount ?? 0;
    const next: TrainingProgressEntry = {
      labelKind,
      labelPhase: labelPhase ?? existing?.labelPhase ?? "labels",
      collected: Math.max(0, Math.min(total, Number.isFinite(collected) ? collected ?? 0 : existing?.collected ?? 0)),
      sampleCount: total,
      labelWorkers: labelWorkers ?? existing?.labelWorkers ?? 1
    };
    trainingProgressState.set(labelKind, next);
    renderTrainingProgress();
  }

  function renderTrainingProgress(): void {
    const entries = Array.from(trainingProgressState.values())
      .sort((left, right) => trainingProgressOrder(left.labelKind) - trainingProgressOrder(right.labelKind));
    trainingProgress.hidden = entries.length === 0;
    trainingProgress.replaceChildren(...entries.map((entry) => {
      const row = document.createElement("div");
      row.className = "training-progress-row";
      const label = document.createElement("span");
      label.className = "training-progress-label";
      label.textContent = trainingProgressLabel(entry.labelKind);
      const progress = document.createElement("progress");
      progress.max = entry.sampleCount;
      progress.value = entry.collected;
      progress.setAttribute("aria-label", `${label.textContent} samples`);
      const detail = document.createElement("span");
      detail.className = "training-progress-detail";
      const workers = `${entry.labelWorkers} worker${entry.labelWorkers === 1 ? "" : "s"}`;
      detail.textContent = `${entry.collected}/${entry.sampleCount} ${trainingProgressPhase(entry)} · ${workers}`;
      row.append(label, progress, detail);
      return row;
    }));
  }

  function trainingProgressOrder(labelKind: LabelKind): number {
    return { search: 0, cpu: 1, duel: 2, "cpu-train": 3, outcome: 4, distilled: 5 }[labelKind] ?? 6;
  }

  function trainingProgressLabel(labelKind: LabelKind): string {
    return {
      search: "GPU Search",
      cpu: "CPU Heuristic",
      duel: "CPU vs GPU",
      "cpu-train": "CPU Training",
      outcome: "Self-play",
      distilled: "Distill"
    }[labelKind] ?? capitalize(labelKind);
  }

  function trainingProgressPhase(entry: TrainingProgressEntry): string {
    if ((entry.labelKind === "search" || entry.labelKind === "cpu") && entry.labelPhase === "positions") {
      return "positions";
    }
    if (entry.labelKind === "search" || entry.labelKind === "cpu") {
      return "labels";
    }
    return "samples";
  }

  function openTrainingModal(): void {
    if (!trainingEnabled) {
      return;
    }
    trainingModal.hidden = false;
    openTrainingButton.setAttribute("aria-expanded", "true");
    renderTrainingButtons();
  }

  function closeTrainingModal(): void {
    trainingModal.hidden = true;
    openTrainingButton.setAttribute("aria-expanded", "false");
  }

  async function startFrontendTraining(): Promise<void> {
    if (!trainingEnabled || trainingRunning) {
      return;
    }
    trainingRunning = true;
    trainingCycle = 0;
    resetTrainingProgress();
    renderTrainingButtons();
    runFrontendTrainingCycle();
  }

  function runFrontendTrainingCycle(): void {
    if (!trainingRunning) {
      return;
    }
    const config = trainingConfig();
    const cycle = trainingCycle + 1;
    setTrainingStatus({
      title: "Collecting",
      tone: "active",
      metrics: filterMetrics([trainingMetric("Run", cycle)])
    });
    if (config.trainingSubject === "cpu") {
      setTrainingStatus({
        title: "Training CPU",
        tone: "active",
        metrics: filterMetrics([
          trainingMetric("Depth", config.cpuDepth),
          trainingMetric("Nodes", config.cpuNodes),
          trainingMetric("Workers", config.cpuWorkers)
        ])
      });
    }
    resetTrainingProgress();
    try {
      const id = ++trainingRequestId;
      const request: TrainingWorkerRequest = {
        id,
        type: "train",
        game: getGame(),
        config
      };
      ensureTrainingWorker().postMessage(request);
    } catch (error) {
      trainingRunning = false;
      setTrainingStatus({
        title: "Training Error",
        tone: "error",
        metrics: filterMetrics([trainingMetric("Message", errorMessage(error))])
      });
      renderTrainingButtons();
    }
  }

  function stopFrontendTraining(): void {
    if (!trainingRunning) {
      return;
    }
    trainingRequestId += 1;
    trainingWorker?.terminate();
    trainingWorker = null;
    trainingRunning = false;
    setTrainingStatus({ title: "Stopped", tone: "warn" });
    resetTrainingProgress();
    renderTrainingButtons();
  }

  async function replaceActiveModel(model: ArrayBuffer): Promise<TrainingReplacementPayload> {
    const response = await fetch("/api/training/model", {
      method: "PUT",
      headers: { "content-type": "application/octet-stream" },
      body: model
    });
    const payload = await readJsonResponse(response);
    if (!response.ok) {
      throw new Error(recordString(payload, "error") ?? `Failed to replace model (${response.status})`);
    }
    resetAiWorker();
    await loadTrainingStatus();
    return isRecord(payload) ? payload as TrainingReplacementPayload : {};
  }

  async function replaceCpuParameters(parameters: string): Promise<TrainingReplacementPayload> {
    const response = await fetch("/api/training/cpu-parameters", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: parameters
    });
    const payload = await readJsonResponse(response);
    if (!response.ok) {
      throw new Error(recordString(payload, "error") ?? `Failed to replace CPU parameters (${response.status})`);
    }
    resetAiWorker();
    await loadTrainingStatus();
    return isRecord(payload) ? payload as TrainingReplacementPayload : {};
  }

  async function readJsonResponse(response: Response): Promise<unknown> {
    const text = await response.text();
    if (!text) {
      return null;
    }
    try {
      return JSON.parse(text);
    } catch {
      return { error: text };
    }
  }

  async function handleTrainingWorkerMessage(event: MessageEvent<TrainingWorkerMessage>): Promise<void> {
    const data = event.data;
    const {
      id,
      ok,
      error,
      model,
      cpuParameters,
      cpuScore,
      loss,
      epoch,
      collected,
      sampleCount,
      labelWorkers,
      labelKind,
      labelPhase,
      selfPlayWorkers,
      searchWorkers,
      bufferSize,
      labelCounts,
      batchSize,
      batchesPerSubmit,
      validationInterval,
      replaySize,
      validationLoss,
      bestValidationLoss,
      epochsWithoutImprovement,
      earlyStopReason,
      nonZeroWeights,
      gpuPhase,
      metrics,
      lossLogValidation
    } = data;
    if (id !== trainingRequestId) {
      return;
    }
    if (!ok) {
      trainingRunning = false;
      setTrainingStatus({
        title: "Training Error",
        tone: "error",
        metrics: filterMetrics([trainingMetric("Message", error)])
      });
      resetTrainingProgress();
      renderTrainingButtons();
      return;
    }
    if (labelWorkers !== undefined && collected === undefined) {
      updateTrainingProgress({ labelKind, labelPhase, collected: 0, sampleCount, labelWorkers });
      return;
    }
    if (cpuParameters) {
      trainingCycle += 1;
      setTrainingStatus({
        title: "Replacing CPU",
        tone: "active",
        metrics: filterMetrics([
          trainingMetric("Run", trainingCycle),
          trainingMetric("Score", cpuScore?.toFixed(2))
        ])
      });
      let replacement: TrainingReplacementPayload | null = null;
      try {
        replacement = await replaceCpuParameters(cpuParameters);
      } catch (replaceError) {
        trainingRunning = false;
        setTrainingStatus({
          title: "CPU Save Error",
          tone: "error",
          metrics: filterMetrics([trainingMetric("Message", errorMessage(replaceError))])
        });
        resetTrainingProgress();
        renderTrainingButtons();
        return;
      }
      setTrainingStatus({
        title: replacement.changed === false ? "CPU Unchanged" : "CPU Saved",
        tone: replacement.changed === false ? "warn" : "ready",
        metrics: filterMetrics([
          trainingMetric("Run", trainingCycle),
          trainingMetric("Hash", compactHash(replacement.newHash), replacement.newHash),
          trainingMetric("Score", cpuScore?.toFixed(2))
        ])
      });
      setTimeout(runFrontendTrainingCycle, 0);
      return;
    }
    if (model) {
      trainingCycle += 1;
      setTrainingStatus({
        title: "Replacing Model",
        tone: "active",
        metrics: filterMetrics([
          trainingMetric("Run", trainingCycle),
          trainingMetric("Loss", formatLoss(loss)),
          trainingMetric("Best Val", formatLoss(bestValidationLoss)),
          trainingMetric("Replay", replayStatusText(metrics?.lossLogValidation)),
          trainingMetric("Stop", earlyStopReason)
        ])
      });
      logTrainingMetrics(trainingCycle, metrics);
      let replacement: TrainingReplacementPayload | null = null;
      try {
        replacement = await replaceActiveModel(model);
      } catch (replaceError) {
        trainingRunning = false;
        setTrainingStatus({
          title: "Save Error",
          tone: "error",
          metrics: filterMetrics([trainingMetric("Message", errorMessage(replaceError))])
        });
        resetTrainingProgress();
        renderTrainingButtons();
        return;
      }
      if (!trainingRunning) {
        renderTrainingButtons();
        return;
      }
      const logValidation = await validateTrainingLossLogs(trainingConfig());
      if (metrics && logValidation) {
        metrics.lossLogValidation = logValidation;
      }
      if (!trainingRunning) {
        renderTrainingButtons();
        return;
      }
      if (replacement.changed === false) {
        const path = replacement.resolvedModelPath ?? replacement.modelPath ?? "model file";
        const hash = replacement.newHash ?? "same hash";
        setTrainingStatus({
          title: "Model Unchanged",
          tone: "warn",
          metrics: filterMetrics([
            trainingMetric("Run", trainingCycle),
            trainingMetric("Hash", compactHash(hash), hash),
            trainingMetric("Path", compactPath(path), path),
            trainingMetric("Replay", replayStatusText(logValidation))
          ])
        });
        setTimeout(runFrontendTrainingCycle, 0);
        return;
      }
      setTrainingStatus({
        title: "Model Saved",
        tone: "ready",
        metrics: filterMetrics([
          trainingMetric("Run", trainingCycle),
          trainingMetric("Weights", nonZeroWeights ?? 0),
          trainingMetric("Hash", compactHash(replacement.newHash), replacement.newHash),
          trainingMetric("Replay", replayStatusText(logValidation))
        ])
      });
      setTimeout(runFrontendTrainingCycle, 0);
      return;
    }
    if (lossLogValidation) {
      const validation = data.lossLogValidation ?? data.validation;
      setTrainingStatus({
        title: "Replaying Loss Logs",
        tone: validation?.failed ? "warn" : "active",
        metrics: filterMetrics([
          trainingMetric("Run", trainingCycle + 1),
          trainingMetric("Changed", validation ? `${validation.changed}/${validation.checked}` : null),
          trainingMetric("Skipped", validation?.skipped)
        ])
      });
      return;
    }
    if (collected !== undefined) {
      updateTrainingProgress({ labelKind, labelPhase, collected, sampleCount, labelWorkers });
      return;
    }
    if (gpuPhase) {
      resetTrainingProgress();
      setTrainingStatus({
        title: "GPU Training",
        tone: "active",
        metrics: filterMetrics([
          trainingMetric("Replay", bufferSize),
          trainingMetric("Batch", batchSize),
          trainingMetric("Self-play", selfPlayWorkers),
          trainingMetric("Search", searchWorkers),
          trainingMetric("Temp", formatTemperature(trainingConfig().explorationTemperature)),
          trainingMetric("Sources", formatLabelCounts(labelCounts))
        ])
      });
      return;
    }
    if (epoch !== undefined) {
      setTrainingStatus({
        title: "Optimizing",
        tone: "active",
        metrics: filterMetrics([
          trainingMetric("Epoch", epoch),
          trainingMetric("Train", formatLoss(loss)),
          trainingMetric("Validation", formatLoss(validationLoss)),
          trainingMetric("Best", formatLoss(bestValidationLoss)),
          trainingMetric("Stale", epochsWithoutImprovement),
          trainingMetric("Replay", replaySize ?? "?"),
          trainingMetric("Batch", batchSize ?? "?"),
          trainingMetric("Submit", batchesPerSubmit),
          trainingMetric("Val every", validationInterval)
        ])
      });
    }
  }

  function validateTrainingLossLogs(config: TrainingConfig): Promise<LossLogValidation | null> {
    if (!trainingRunning || (config.lossLogReplay ?? 0) <= 0) {
      return Promise.resolve(null);
    }
    return new Promise((resolve) => {
      const worker = new Worker("./training-worker.js", { type: "module" });
      const id = ++trainingRequestId;
      const cleanup = (): void => {
        worker.removeEventListener("message", handleMessage);
        worker.removeEventListener("error", handleError);
        worker.removeEventListener("messageerror", handleError);
        worker.terminate();
      };
      const handleMessage = (event: MessageEvent<TrainingWorkerMessage>): void => {
        if (event.data.id !== id) {
          return;
        }
        if (event.data.type === "lossLogValidation") {
          cleanup();
          resolve(event.data.validation ?? null);
          return;
        }
        if (event.data.lossLogValidation) {
          setTrainingStatus({
            title: "Replaying Loss Logs",
            tone: event.data.lossLogValidation.failed ? "warn" : "active",
            metrics: filterMetrics([
              trainingMetric("Run", trainingCycle),
              trainingMetric("Changed", `${event.data.lossLogValidation.changed}/${event.data.lossLogValidation.checked}`)
            ])
          });
        }
        if (event.data.ok === false) {
          cleanup();
          resolve(null);
        }
      };
      const handleError = (): void => {
        cleanup();
        resolve(null);
      };
      worker.addEventListener("message", handleMessage);
      worker.addEventListener("error", handleError);
      worker.addEventListener("messageerror", handleError);
      const request: TrainingWorkerRequest = {
        id,
        type: "validateLossLogs",
        config
      };
      worker.postMessage(request);
    });
  }

  function logTrainingMetrics(run: number, metrics: TrainingRunMetrics | undefined): void {
    if (!metrics?.phases) {
      return;
    }
    console.groupCollapsed(`Training run ${run}: ${metrics.totalMs ?? "?"} ms`);
    console.table(Object.entries(metrics.phases).map(([phase, ms]) => ({
      phase,
      ms
    })));
    if (metrics.sampleRates && Object.keys(metrics.sampleRates).length > 0) {
      console.table(Object.entries(metrics.sampleRates).map(([source, samplesPerSecond]) => ({
        source,
        samplesPerSecond
      })));
    }
    if (metrics.lossLogValidation) {
      console.table([metrics.lossLogValidation]);
    }
    console.groupEnd();
  }

  function replayStatusText(validation: LossLogValidation | null | undefined): string | null {
    if (!validation || validation.checked === 0) {
      return null;
    }
    const label = validation.failed ? "No changes" : "Changed";
    return `${label} ${validation.changed}/${validation.checked}`;
  }

  function formatLoss(loss: number | undefined): string {
    return Number.isFinite(loss) ? (loss as number).toFixed(2) : "pending";
  }

  function formatTemperature(value: number): string {
    return Number.isFinite(value) ? value.toFixed(2).replace(/\.?0+$/, "") : "?";
  }

  function formatLabelCounts(counts: Record<string, number> | undefined): string {
    if (!counts || typeof counts !== "object") {
      return "No source counts";
    }
    return ["search", "outcome", "distilled", "unknown"]
      .filter((key) => counts[key])
      .map((key) => `${key} ${counts[key]}`)
      .join(", ") || "No source counts";
  }

  return {
    loadStatus: loadTrainingStatus,
    renderButtons: renderTrainingButtons,
    openModal: openTrainingModal,
    closeModal: closeTrainingModal,
    start: startFrontendTraining,
    stop: stopFrontendTraining
  };
}

function requireElement<T extends Element>(element: T | null, id: string): T {
  if (!element) {
    throw new Error(`Missing required #${id} element.`);
  }
  return element;
}

function filterMetrics(metrics: Array<TrainingMetric | null>): TrainingMetric[] {
  return metrics.filter((metric): metric is TrainingMetric => Boolean(metric));
}

function gpuAdapterName(adapter: GPUAdapter): string {
  const info = (adapter as GPUAdapter & { info?: { description?: string; device?: string; vendor?: string } }).info;
  return info?.description || info?.device || info?.vendor || "WebGPU";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function recordString(value: unknown, key: string): string | null {
  if (!isRecord(value)) {
    return null;
  }
  const field = value[key];
  return typeof field === "string" ? field : null;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
