import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";

const url = optionValue("--url") ?? process.env.CHRONOFISH_URL ?? "http://127.0.0.1:5173";
const benchmarkIterations = positiveNumber(optionValue("--benchmark-iterations") ?? process.env.CHRONOFISH_GPU_SMOKE_ITERATIONS, 3, 1);
const benchmarkTimeMs = positiveNumber(optionValue("--benchmark-time-ms") ?? process.env.CHRONOFISH_GPU_SMOKE_TIME_MS, 5_000, 1000);
const smokeTimeoutMs = positiveNumber(optionValue("--smoke-timeout-ms") ?? process.env.CHRONOFISH_GPU_SMOKE_TIMEOUT_MS, 45_000, 5_000);
const runPerformanceGates = !hasFlag("--skip-performance-gates");
const smokeGpuMode = optionValue("--gpu-mode") ?? "full";
const forceFullGpu = smokeGpuMode === "full" && !hasFlag("--allow-full-fallback");
const expectFullGpu = smokeGpuMode === "full" && forceFullGpu;
const allowSoftwareAdapter = hasFlag("--allow-software-adapter");
const disableNeural = hasFlag("--disable-neural");
const cpuMinDepthOnly = hasFlag("--cpu-min-depth-only");
const trainingBenchmarkSubject = optionValue("--training-benchmark");
const trainingBenchmarkTimeoutMs = positiveNumber(optionValue("--training-timeout-ms"), 180_000, 10_000);
const trainingBenchmarkSamples = positiveNumber(optionValue("--training-samples"), 16, 1);
const trainingBenchmarkEpochs = positiveNumber(optionValue("--training-epochs"), 256, 1);
const trainingBenchmarkBatch = positiveNumber(optionValue("--training-batch"), 64, 16);
const browser = process.env.CHRONOFISH_BROWSER ?? await findBrowser();
if (!browser) {
  throw new Error("Set CHRONOFISH_BROWSER to a Chrome/Chromium binary for GPU frontier smoke tests.");
}
if (typeof WebSocket !== "function") {
  throw new Error("This smoke test requires a Node.js runtime with global WebSocket support.");
}

const fixtures = [
  { name: "one-board-initial", game: initialGame(), depth: 1, nodes: 64 },
  { name: "three-board-present", game: multiBoardGame(3), depth: 2, nodes: 256 },
  { name: "five-board-present", game: multiBoardGame(5), depth: 2, nodes: 512 },
  { name: "forced-multi-move-turn", game: multiBoardGame(3), depth: 1, nodes: 256, minMoves: 2 },
  {
    name: "historical-branch",
    game: historicalBranchGame(),
    depth: 1,
    nodes: 256,
    expectedTarget: move({ timelineId: 0, time: 1, x: 3, y: 3 }, { timelineId: 0, time: 0, x: 3, y: 3 })
  },
  {
    name: "capture",
    game: captureGame(),
    depth: 1,
    nodes: 128,
    expectedTarget: move({ timelineId: 0, time: 0, x: 0, y: 0 }, { timelineId: 0, time: 0, x: 0, y: 3 })
  },
  {
    name: "castling",
    game: castlingGame(),
    depth: 1,
    nodes: 128,
    expectedTarget: move({ timelineId: 0, time: 0, x: 4, y: 0 }, { timelineId: 0, time: 0, x: 6, y: 0 })
  },
  {
    name: "en-passant",
    game: enPassantGame(),
    depth: 1,
    nodes: 128,
    expectedTarget: move({ timelineId: 0, time: 0, x: 4, y: 4 }, { timelineId: 0, time: 0, x: 3, y: 5 })
  },
  {
    name: "promotion",
    game: promotionGame(),
    depth: 1,
    nodes: 128,
    expectedTarget: move({ timelineId: 0, time: 0, x: 0, y: 6 }, { timelineId: 0, time: 0, x: 0, y: 7 })
  },
  {
    name: "terminal-pressure",
    game: terminalPressureGame(),
    depth: 1,
    nodes: 128,
    expectedTarget: move({ timelineId: 0, time: 0, x: 4, y: 6 }, { timelineId: 0, time: 0, x: 4, y: 7 })
  }
];

async function main() {
  const userDataDir = await mkdtemp(path.join(os.tmpdir(), "chronofish-gpu-smoke-"));
  const child = spawn(browser, [
    "--headless=new",
    "--disable-gpu-sandbox",
    "--enable-unsafe-webgpu",
    "--use-angle=vulkan",
    "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
    "--remote-debugging-port=0",
    "--user-data-dir=" + userDataDir,
    "about:blank"
  ], { stdio: ["ignore", "ignore", "pipe"] });

  try {
    const endpoint = await devtoolsEndpoint(child);
    const cdp = await CdpSession.connect(endpoint);
    const target = await cdp.send("Target.createTarget", { url });
    const attached = await cdp.send("Target.attachToTarget", { targetId: target.targetId, flatten: true });
    const sessionId = attached.sessionId;
    await cdp.send("Page.enable", {}, sessionId);
    await cdp.send("Runtime.enable", {}, sessionId);
    await cdp.send("Page.navigate", { url }, sessionId);
    await delay(1000);
    if (cpuMinDepthOnly) {
      const cpu = await cdp.evaluate(cpuMinimumDepthExpression(), sessionId);
      if (!cpu.ok) {
        throw new Error("CPU minimum-depth smoke failed: " + (cpu.error ?? "unknown error"));
      }
      console.log(JSON.stringify(cpu.result));
      cdp.close();
      return;
    }
    const adapter = await cdp.evaluate(adapterInfoExpression(), sessionId);
    console.log(JSON.stringify({ fixture: "adapter", ...adapter }));
    if (smokeGpuMode === "full" && adapter.fallback && !allowSoftwareAdapter) {
      throw new Error("Full GPU smoke selected a fallback software adapter. Install/configure Chromium with native Vulkan WebGPU support, or pass --allow-software-adapter for a deliberately slow functional run.");
    }
    if (trainingBenchmarkSubject) {
      if (trainingBenchmarkSubject !== "gpu" && trainingBenchmarkSubject !== "cpu") {
        throw new Error("--training-benchmark must be gpu or cpu.");
      }
      const benchmark = await cdp.evaluate(trainingBenchmarkExpression(trainingBenchmarkSubject), sessionId);
      if (!benchmark.ok) {
        throw new Error("training benchmark failed: " + (benchmark.error ?? "unknown error"));
      }
      console.log(JSON.stringify(benchmark.result));
      cdp.close();
      return;
    }

    for (const fixture of fixtures) {
      if (fixture.expectedTarget) {
        const legal = await cdp.evaluate(legalTargetExpression(fixture), sessionId);
        if (!legal.ok) {
          throw new Error(fixture.name + " legal-target preflight failed: " + (legal.error ?? "unknown error"));
        }
        console.log(JSON.stringify(legal.result));
      }
      const result = await cdp.evaluate(workerSmokeExpression(fixture), sessionId);
      if (!result.ok) {
        throw new Error(fixture.name + " failed: " + (result.error ?? "unknown error"));
      }
      const search = result.result;
      if (search.status !== "ok" || !Array.isArray(search.moves) || search.moves.length === 0) {
        throw new Error(fixture.name + " returned no legal full-mode turn: " + JSON.stringify(search));
      }
      if (search.authoritativeReplay !== true) {
        throw new Error(fixture.name + " did not report authoritative WASM replay validation.");
      }
      if (smokeGpuMode === "full" && !forceFullGpu && search.gpuMode !== "hybrid") {
        throw new Error(fixture.name + " did not report automatic hybrid fallback: " + JSON.stringify(search));
      }
      if (fixture.minMoves && search.moves.length < fixture.minMoves) {
        throw new Error(fixture.name + " did not complete the expected multi-board turn: " + JSON.stringify(search.moves));
      }
      if (expectFullGpu && (!search.gpuDiagnostics || search.gpuDiagnostics.readbacks !== 1)) {
        throw new Error(fixture.name + " did not report the expected single-readback frontier diagnostics.");
      }
      if (expectFullGpu && search.gpuSearch !== "neural-frontier" && search.gpuSearch !== "heuristic-frontier") {
        throw new Error(fixture.name + " did not use the resident frontier path: " + JSON.stringify(search));
      }
      if (expectFullGpu && search.gpuDiagnostics.candidateOverflow) {
        throw new Error(fixture.name + " reported a capacity-truncated frontier: " + JSON.stringify(search.gpuDiagnostics));
      }
      if (expectFullGpu && (search.gpuDiagnostics.nodes ?? 0) > 0 && search.gpuDiagnostics.selectedCount < Math.min(search.gpuDiagnostics.frontierWidth ?? 0, search.gpuDiagnostics.nodes ?? 0)) {
        throw new Error(fixture.name + " underfilled the resident frontier: " + JSON.stringify(search.gpuDiagnostics));
      }
      console.log(JSON.stringify({
        fixture: fixture.name,
        moves: search.moves.length,
        depth: search.depth,
        nodes: search.nodes,
        gpuSearch: search.gpuSearch,
        diagnostics: search.gpuDiagnostics
      }));
    }
    const stale = await cdp.evaluate(staleGenerationExpression(multiBoardGame(5), initialGame()), sessionId);
    if (!stale.ok) {
      throw new Error("stale-generation failed: " + (stale.error ?? "unknown error"));
    }
    console.log(JSON.stringify(stale.result));
    const deviceLoss = await cdp.evaluate(deviceLossExpression(initialGame()), sessionId);
    if (!deviceLoss.ok) {
      throw new Error("device-loss failed: " + (deviceLoss.error ?? "unknown error"));
    }
    console.log(JSON.stringify(deviceLoss.result));
    if (runPerformanceGates) {
      const benchmark = await cdp.evaluate(benchmarkExpression(benchmarkIterations, benchmarkTimeMs), sessionId);
      if (!benchmark.ok) {
        throw new Error("performance-gates failed: " + (benchmark.error ?? "unknown error"));
      }
      console.log(JSON.stringify(benchmark.result));
    }
    cdp.close();
  } finally {
    child.kill("SIGTERM");
    await rm(userDataDir, { recursive: true, force: true });
  }
}

function trainingBenchmarkExpression(subject) {
  const config = subject === "gpu"
    ? {
        trainingSubject: "gpu",
        trainingModes: ["distill"],
        samples: trainingBenchmarkSamples,
        selfPlayWorkers: 1,
        searchWorkers: 1,
        explorationTemperature: 0,
        depth: 1,
        nodes: 64,
        learningRate: 0.005,
        epochs: trainingBenchmarkEpochs,
        maxBuffer: Math.max(16, trainingBenchmarkSamples),
        batchSize: trainingBenchmarkBatch,
        validationSplit: 0.2,
        validationInterval: Math.min(64, trainingBenchmarkEpochs),
        patience: 2,
        weightDecay: 0.00001,
        lossLogReplay: 0
      }
    : {
        trainingSubject: "cpu",
        trainingModes: ["vsCpu"],
        samples: Math.min(trainingBenchmarkSamples, 8),
        depth: 1,
        nodes: 16,
        cpuDepth: 1,
        cpuNodes: 16,
        cpuTrainingTimeMs: 20,
        cpuCandidates: 2,
        cpuFinalists: 1,
        cpuPairBatch: 1,
        cpuOpponentVariants: 1,
        cpuScreeningOpponentVariants: 1,
        cpuRoundsPerVariant: 1,
        cpuHallOfFameEntries: 0,
        cpuLeagueContenders: 1,
        cpuLeagueHallOfFameEntries: 0,
        cpuMinPairs: 1,
        cpuMaxPairs: 1,
        cpuDrawWindow: 4,
        cpuDrawRateLimit: 0.8,
        cpuMaxMatchPlies: 1,
        cpuMaxMatchTimeMs: 3_000,
        cpuMaxGenerationsWithoutCandidate: 1,
        cpuWorkers: 1,
        cpuTrainSeconds: 10
      };
  const payload = {
    id: "training-benchmark",
    type: "train",
    game: initialGame(),
    config
  };
  return "new Promise((resolve) => {"
    + "const started = performance.now();"
    + "let last = null;"
    + "const worker = new Worker('./training-worker.js', { type: 'module' });"
    + "const timeout = setTimeout(() => { worker.terminate(); resolve({ ok: false, error: 'training benchmark timed out: ' + JSON.stringify(last) }); }, "
    + trainingBenchmarkTimeoutMs + ");"
    + "worker.onmessage = (event) => {"
    + "const data = event.data || {};"
    + "if (data.id !== 'training-benchmark') return;"
    + "last = { labelKind: data.labelKind ?? null, collected: data.collected ?? null, sampleCount: data.sampleCount ?? null, generation: data.generation ?? null, metrics: data.metrics ?? null };"
    + "if (data.ok === false) { clearTimeout(timeout); worker.terminate(); resolve({ ok: false, error: data.error || 'training worker error' }); return; }"
    + "if (!data.model && !data.cpuParameters) return;"
    + "clearTimeout(timeout); worker.terminate();"
    + "resolve({ ok: true, result: {"
    + "fixture: 'training-benchmark', subject: " + JSON.stringify(subject) + ", elapsedMs: Math.round(performance.now() - started),"
    + "metrics: data.metrics || null, loss: data.loss ?? null, initialValidationLoss: data.initialValidationLoss ?? null,"
    + "bestValidationLoss: data.bestValidationLoss ?? null, initialPolicyValidationLoss: data.initialPolicyValidationLoss ?? null,"
    + "bestPolicyValidationLoss: data.bestPolicyValidationLoss ?? null, valueCheckpointImproved: data.valueCheckpointImproved ?? null,"
    + "policyCheckpointImproved: data.policyCheckpointImproved ?? null, modelChanged: data.modelChanged ?? null,"
    + "replaySize: data.replaySize ?? null, trainingSampleCount: data.trainingSampleCount ?? null, policyTrainingSampleCount: data.policyTrainingSampleCount ?? null, cpuScore: data.cpuScore ?? null,"
    + "phasePercentages: data.metrics && data.metrics.phases && data.metrics.totalMs ? Object.fromEntries(Object.entries(data.metrics.phases).map(([name, ms]) => [name, Number((ms * 100 / data.metrics.totalMs).toFixed(1))])) : null,"
    + "trainingSamplesPerSecond: data.trainingSampleCount && data.metrics && data.metrics.phases && data.metrics.phases.train ? Number((data.trainingSampleCount / (data.metrics.phases.train / 1000)).toFixed(2)) : null"
    + "} });"
    + "};"
    + "worker.onerror = (event) => { clearTimeout(timeout); worker.terminate(); resolve({ ok: false, error: event.message || 'training worker error' }); };"
    + "worker.postMessage(" + JSON.stringify(payload) + ");"
    + "})";
}

function cpuMinimumDepthExpression() {
  const payload = {
    id: "cpu-minimum-depth",
    game: initialGame(),
    depth: 3,
    minDepth: 3,
    nodes: 20_000,
    timeMs: 1,
    searchStrategy: "alpha-beta",
    partitionIndex: 0
  };
  return "new Promise((resolve) => {"
    + "const worker = new Worker('./cpu-ai-worker.js', { type: 'module' });"
    + "const timeout = setTimeout(() => { worker.terminate(); resolve({ ok: false, error: 'timed out waiting for CPU minimum-depth worker' }); }, 30000);"
    + "worker.onmessage = (event) => {"
    + "clearTimeout(timeout); worker.terminate();"
    + "const data = event.data; const result = data && data.result;"
    + "resolve(data && data.ok && result && result.status === 'ok' && result.depth >= 3 && Array.isArray(result.moves) && result.moves.length > 0"
    + " ? { ok: true, result: { fixture: 'cpu-minimum-depth', depth: result.depth, moves: result.moves.length, nodes: result.nodes } }"
    + " : { ok: false, error: JSON.stringify(data) });"
    + "};"
    + "worker.onerror = (event) => { clearTimeout(timeout); worker.terminate(); resolve({ ok: false, error: event.message || 'CPU worker error' }); };"
    + "worker.postMessage(" + JSON.stringify(payload) + ");"
    + "})";
}

function adapterInfoExpression() {
  return "(async () => {"
    + "if (!navigator.gpu) return { available: false };"
    + "const adapter = await navigator.gpu.requestAdapter();"
    + "if (!adapter) return { available: false };"
    + "const info = adapter.info || {};"
    + "return { available: true, vendor: info.vendor || '', architecture: info.architecture || '', device: info.device || '', description: info.description || '', fallback: Boolean(info.isFallbackAdapter) };"
    + "})()";
}

function legalTargetExpression(fixture) {
  const payload = {
    id: fixture.name + "-legal-targets",
    type: "legalTargets",
    game: fixture.game,
    position: fixture.expectedTarget.from
  };
  const target = fixture.expectedTarget.to;
  return "new Promise((resolve) => {"
    + "const worker = new Worker('./ai-worker.js', { type: 'module' });"
    + "const timeout = setTimeout(() => { worker.terminate(); resolve({ ok: false, error: 'timed out waiting for legal target preflight' }); }, 15000);"
    + "worker.onmessage = (event) => {"
    + "clearTimeout(timeout);"
    + "worker.terminate();"
    + "const data = event.data;"
    + "const target = " + JSON.stringify(target) + ";"
    + "const targets = data && data.ok && data.selection && Array.isArray(data.selection.targets) ? data.selection.targets : [];"
    + "const matched = targets.some((candidate) => candidate.timelineId === target.timelineId && candidate.time === target.time && candidate.x === target.x && candidate.y === target.y);"
    + "resolve(matched ? { ok: true, result: { fixture: " + JSON.stringify(fixture.name) + ", preflight: 'legal-target', targetCount: targets.length } } : { ok: false, error: 'expected target missing from GPU legal targets: ' + JSON.stringify({ data, target }) });"
    + "};"
    + "worker.onerror = (event) => { clearTimeout(timeout); worker.terminate(); resolve({ ok: false, error: event.message || 'worker error' }); };"
    + "worker.postMessage(" + JSON.stringify(payload) + ");"
    + "})";
}

function workerSmokeExpression(fixture) {
  const payload = {
    id: fixture.name,
    game: fixture.game,
    depth: fixture.depth,
    nodes: fixture.nodes,
    timeMs: 15_000,
    gpuMode: smokeGpuMode,
    forceFullGpu,
    disableNeural,
    randomSeed: 12345
  };
  return singleSearchExpression(payload, smokeTimeoutMs);
}

function move(from, to) {
  return { from, to };
}

function staleGenerationExpression(slowGame, fastGame) {
  const first = {
    id: "stale-slow",
    game: slowGame,
    depth: 3,
    nodes: 1024,
    timeMs: 15_000,
    gpuMode: smokeGpuMode,
    forceFullGpu,
    disableNeural,
    randomSeed: 12345
  };
  const second = {
    id: "stale-fast",
    game: fastGame,
    depth: 1,
    nodes: 64,
    timeMs: 15_000,
    gpuMode: smokeGpuMode,
    forceFullGpu,
    disableNeural,
    randomSeed: 12345
  };
  return "new Promise((resolve) => {"
    + "const worker = new Worker('./ai-worker.js', { type: 'module' });"
    + "const messages = [];"
    + "const timeout = setTimeout(() => { worker.terminate(); resolve({ ok: false, error: 'timed out waiting for stale-generation check: ' + JSON.stringify(messages) }); }, 30000);"
    + "worker.onmessage = (event) => {"
    + "messages.push(event.data);"
    + "if (event.data && event.data.id === 'stale-fast') {"
    + "setTimeout(() => {"
    + "clearTimeout(timeout);"
    + "worker.terminate();"
    + "const stalePublished = messages.some((message) => message && message.id === 'stale-slow');"
    + "resolve(stalePublished ? { ok: false, error: 'stale search published after superseding request' } : { ok: true, result: { fixture: 'stale-generation', messages: messages.map((message) => message.id) } });"
    + "}, 2000);"
    + "}"
    + "};"
    + "worker.onerror = (event) => { clearTimeout(timeout); worker.terminate(); resolve({ ok: false, error: event.message || 'worker error' }); };"
    + "worker.postMessage(" + JSON.stringify(first) + ");"
    + "setTimeout(() => worker.postMessage(" + JSON.stringify(second) + "), 0);"
    + "})";
}

function deviceLossExpression(game) {
  const before = {
    id: "device-loss-before",
    game,
    depth: 1,
    nodes: 64,
    timeMs: 15_000,
    gpuMode: smokeGpuMode,
    forceFullGpu,
    disableNeural,
    randomSeed: 12345
  };
  const after = { ...before, id: "device-loss-after" };
  const lose = { id: "device-loss-trigger", type: "debugLoseDevice" };
  return "new Promise((resolve) => {"
    + "const worker = new Worker('./ai-worker.js', { type: 'module' });"
    + "const messages = [];"
    + "let step = 0;"
    + "const timeout = setTimeout(() => { worker.terminate(); resolve({ ok: false, error: 'timed out waiting for device-loss check: ' + JSON.stringify(messages) }); }, 45000);"
    + "function fail(error) { clearTimeout(timeout); worker.terminate(); resolve({ ok: false, error }); }"
    + "function assertFrontier(message, label) {"
    + "const needsFrontierDiagnostics = " + JSON.stringify(expectFullGpu) + ";"
    + "if (!message.ok || !message.result || message.result.status !== 'ok' || message.result.authoritativeReplay !== true || (needsFrontierDiagnostics && (!message.result.gpuDiagnostics || message.result.gpuDiagnostics.readbacks !== 1 || message.result.gpuDiagnostics.candidateOverflow || message.result.gpuDiagnostics.selectedCount < Math.min(message.result.gpuDiagnostics.frontierWidth || 0, message.result.gpuDiagnostics.nodes || 0) || (message.result.gpuSearch !== 'neural-frontier' && message.result.gpuSearch !== 'heuristic-frontier')))) {"
    + "fail(label + ' did not return resident frontier diagnostics: ' + JSON.stringify(message));"
    + "return false;"
    + "}"
    + "return true;"
    + "}"
    + "worker.onmessage = (event) => {"
    + "const message = event.data;"
    + "messages.push(message);"
    + "if (step === 0) { if (!assertFrontier(message, 'before device loss')) return; step = 1; worker.postMessage(" + JSON.stringify(lose) + "); return; }"
    + "if (step === 1) { if (!message.ok || message.lostDevice !== true) { fail('worker did not destroy a cached device: ' + JSON.stringify(message)); return; } step = 2; worker.postMessage(" + JSON.stringify(after) + "); return; }"
    + "if (step === 2) { if (!assertFrontier(message, 'after device loss')) return; clearTimeout(timeout); worker.terminate(); resolve({ ok: true, result: { fixture: 'device-loss', messages: messages.map((entry) => entry.id) } }); }"
    + "};"
    + "worker.onerror = (event) => fail(event.message || 'worker error');"
    + "worker.postMessage(" + JSON.stringify(before) + ");"
    + "})";
}

function benchmarkExpression(iterations, timeMs) {
  const cases = [
    { name: "one-board-initial", game: initialGame(), depth: 1, nodes: 64, maxRegression: 1.10 },
    { name: "three-board-present", game: multiBoardGame(3), depth: 2, nodes: 256, minSpeedup: 2 },
    { name: "five-board-present", game: multiBoardGame(5), depth: 2, nodes: 512, minSpeedup: 2 }
  ];
  return "(async () => {"
    + "const cases = " + JSON.stringify(cases) + ";"
    + "const iterations = " + JSON.stringify(iterations) + ";"
    + "const timeMs = " + JSON.stringify(timeMs) + ";"
    + "async function run(payload) {"
    + "const started = performance.now();"
    + "const response = await " + singleSearchExpressionSource("payload", "timeMs + 15000") + ";"
    + "const elapsedMs = performance.now() - started;"
    + "if (!response.ok || !response.result || response.result.status !== 'ok' || response.result.authoritativeReplay !== true || !Array.isArray(response.result.moves) || response.result.moves.length === 0) {"
    + "throw new Error(payload.id + ' failed: ' + JSON.stringify(response));"
    + "}"
    + "return { elapsedMs, result: response.result };"
    + "}"
    + "function median(values) { const sorted = values.slice().sort((a, b) => a - b); return sorted[Math.floor(sorted.length / 2)]; }"
    + "const summaries = [];"
    + "for (const entry of cases) {"
    + "const full = [];"
    + "const hybrid = [];"
    + "await run({ id: entry.name + '-warmup-full', game: entry.game, depth: entry.depth, nodes: entry.nodes, timeMs, gpuMode: 'full', randomSeed: 12345 });"
    + "await run({ id: entry.name + '-warmup-hybrid', game: entry.game, depth: entry.depth, nodes: entry.nodes, timeMs, gpuMode: 'hybrid', randomSeed: 12345 });"
    + "for (let index = 0; index < iterations; index += 1) {"
    + "const fullRun = await run({ id: entry.name + '-full-' + index, game: entry.game, depth: entry.depth, nodes: entry.nodes, timeMs, gpuMode: 'full', randomSeed: 12345 });"
    + "const hybridRun = await run({ id: entry.name + '-hybrid-' + index, game: entry.game, depth: entry.depth, nodes: entry.nodes, timeMs, gpuMode: 'hybrid', randomSeed: 12345 });"
    + "if (!fullRun.result.gpuDiagnostics || fullRun.result.gpuDiagnostics.readbacks !== 1 || fullRun.result.gpuDiagnostics.candidateOverflow || fullRun.result.gpuDiagnostics.selectedCount < Math.min(fullRun.result.gpuDiagnostics.frontierWidth || 0, fullRun.result.gpuDiagnostics.nodes || 0) || (fullRun.result.gpuSearch !== 'neural-frontier' && fullRun.result.gpuSearch !== 'heuristic-frontier')) { throw new Error(entry.name + ' full mode did not use filled resident frontier diagnostics'); }"
    + "full.push(fullRun.elapsedMs);"
    + "hybrid.push(hybridRun.elapsedMs);"
    + "}"
    + "const fullMedianMs = median(full);"
    + "const hybridMedianMs = median(hybrid);"
    + "const speedup = hybridMedianMs / Math.max(1, fullMedianMs);"
    + "if (entry.maxRegression && fullMedianMs > hybridMedianMs * entry.maxRegression) { throw new Error(entry.name + ' regressed over hybrid: full=' + fullMedianMs + ' hybrid=' + hybridMedianMs); }"
    + "if (entry.minSpeedup && speedup < entry.minSpeedup) { throw new Error(entry.name + ' did not meet speedup gate: ' + speedup + 'x'); }"
    + "summaries.push({ fixture: entry.name, fullMedianMs: Math.round(fullMedianMs), hybridMedianMs: Math.round(hybridMedianMs), speedup: Math.round(speedup * 100) / 100 });"
    + "}"
    + "return { ok: true, result: { fixture: 'performance-gates', iterations, timeMs, summaries } };"
    + "})().catch((error) => ({ ok: false, error: error instanceof Error ? error.message : String(error) }))";
}

function singleSearchExpression(payload, timeoutMs) {
  return "(() => { const payload = " + JSON.stringify(payload) + "; const timeoutMs = " + JSON.stringify(timeoutMs) + "; return "
    + singleSearchExpressionSource("payload", "timeoutMs")
    + "; })()";
}

function singleSearchExpressionSource(payloadExpression, timeoutExpression) {
  return "new Promise((resolve) => {"
    + "const worker = new Worker('./ai-worker.js', { type: 'module' });"
    + "const timeout = setTimeout(() => { worker.terminate(); resolve({ ok: false, error: 'timed out waiting for AI worker' }); }, " + timeoutExpression + ");"
    + "worker.onmessage = (event) => { clearTimeout(timeout); worker.terminate(); resolve(event.data); };"
    + "worker.onerror = (event) => { clearTimeout(timeout); worker.terminate(); resolve({ ok: false, error: event.message || 'worker error' }); };"
    + "worker.postMessage(" + payloadExpression + ");"
    + "})";
}

class CdpSession {
  static connect(endpoint) {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(endpoint);
      const session = new CdpSession(socket);
      socket.addEventListener("open", () => resolve(session), { once: true });
      socket.addEventListener("error", () => reject(new Error("failed to connect to DevTools endpoint")), { once: true });
    });
  }

  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      const pending = this.pending.get(message.id);
      if (!pending) {
        return;
      }
      this.pending.delete(message.id);
      if (message.error) {
        pending.reject(new Error(message.error.message));
      } else {
        pending.resolve(message.result ?? {});
      }
    });
  }

  send(method, params = {}, sessionId = undefined) {
    const id = this.nextId++;
    const payload = sessionId ? { id, method, params, sessionId } : { id, method, params };
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify(payload));
    });
  }

  async evaluate(expression, sessionId) {
    const result = await this.send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
      userGesture: true
    }, sessionId);
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.text ?? "Runtime.evaluate failed");
    }
    return result.result?.value;
  }

  close() {
    this.socket.close();
  }
}

function devtoolsEndpoint(child) {
  return new Promise((resolve, reject) => {
    let stderr = "";
    const timeout = setTimeout(() => reject(new Error("browser did not expose a DevTools endpoint")), 10_000);
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
      const match = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) {
        clearTimeout(timeout);
        resolve(match[1]);
      }
    });
    child.on("exit", (code) => {
      clearTimeout(timeout);
      reject(new Error("browser exited before DevTools endpoint was ready (" + code + ")"));
    });
  });
}

async function findBrowser() {
  for (const candidate of ["google-chrome", "chromium", "chromium-browser"]) {
    if (await commandExists(candidate)) {
      return candidate;
    }
  }
  return null;
}

function commandExists(command) {
  return new Promise((resolve) => {
    const child = spawn("sh", ["-lc", "command -v " + command], { stdio: "ignore" });
    child.on("exit", (code) => resolve(code === 0));
  });
}

function optionValue(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : null;
}

function hasFlag(name) {
  return process.argv.includes(name);
}

function positiveNumber(value, fallback, minimum) {
  const parsed = Number(value ?? fallback);
  return Number.isFinite(parsed) ? Math.max(minimum, parsed) : fallback;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function initialGame() {
  return {
    turn: "white",
    nextTimelineId: 1,
    nextBlackTimelineId: -1,
    checkedRoyals: [],
    royalCaptureBy: null,
    timelines: [{
      id: 0,
      row: 0,
      label: "T0",
      owner: "neutral",
      boards: [{ time: 0, sideToMove: "white", castling: 15, enPassant: null, origin: null, board: startingBoard() }]
    }]
  };
}

function multiBoardGame(count) {
  const center = Math.floor(count / 2);
  return {
    turn: "white",
    nextTimelineId: center + 1,
    nextBlackTimelineId: -center - 1,
    checkedRoyals: [],
    royalCaptureBy: null,
    timelines: Array.from({ length: count }, (_, index) => {
      const id = index - center;
      const board = emptyBoardWithKings();
      board[1][Math.min(7, index + 1)] = { color: "white", type: "pawn" };
      board[6][Math.max(0, 6 - index)] = { color: "black", type: "pawn" };
      return {
        id,
        row: id,
        label: "T" + id,
        owner: id === 0 ? "neutral" : id > 0 ? "white" : "black",
        boards: [{ time: 0, sideToMove: "white", castling: 0, enPassant: null, origin: null, board }]
      };
    })
  };
}

function historicalBranchGame() {
  const past = emptyBoardWithKings();
  past[3][3] = { color: "black", type: "pawn" };
  const latest = emptyBoardWithKings();
  latest[3][3] = { color: "white", type: "queen" };
  latest[6][6] = { color: "black", type: "pawn" };
  return singleTimelineGame([
    { time: 0, sideToMove: "white", castling: 0, enPassant: null, origin: null, board: past },
    { time: 1, sideToMove: "white", castling: 0, enPassant: null, origin: null, board: latest }
  ]);
}

function captureGame() {
  const board = emptyBoardWithKings();
  board[0][0] = { color: "white", type: "rook" };
  board[3][0] = { color: "black", type: "pawn" };
  board[6][6] = { color: "black", type: "pawn" };
  return singleTimelineGame([{ time: 0, sideToMove: "white", castling: 0, enPassant: null, origin: null, board }]);
}

function castlingGame() {
  const board = emptyBoard();
  board[0][0] = { color: "white", type: "rook" };
  board[0][4] = { color: "white", type: "king" };
  board[0][7] = { color: "white", type: "rook" };
  board[7][4] = { color: "black", type: "king" };
  board[6][3] = { color: "black", type: "pawn" };
  return singleTimelineGame([{ time: 0, sideToMove: "white", castling: 3, enPassant: null, origin: null, board }]);
}

function enPassantGame() {
  const board = emptyBoardWithKings();
  board[4][4] = { color: "white", type: "pawn" };
  board[4][3] = { color: "black", type: "pawn" };
  board[6][6] = { color: "black", type: "pawn" };
  return singleTimelineGame([{
    time: 0,
    sideToMove: "white",
    castling: 0,
    enPassant: { x: 3, y: 5, capturedX: 3, capturedY: 4 },
    origin: null,
    board
  }]);
}

function promotionGame() {
  const board = emptyBoard();
  board[0][4] = { color: "white", type: "king" };
  board[6][0] = { color: "white", type: "pawn" };
  board[7][7] = { color: "black", type: "king" };
  board[6][6] = { color: "black", type: "pawn" };
  return singleTimelineGame([{ time: 0, sideToMove: "white", castling: 0, enPassant: null, origin: null, board }]);
}

function terminalPressureGame() {
  const board = emptyBoard();
  board[0][4] = { color: "white", type: "king" };
  board[6][4] = { color: "white", type: "queen" };
  board[7][4] = { color: "black", type: "king" };
  board[6][6] = { color: "black", type: "pawn" };
  return singleTimelineGame([{ time: 0, sideToMove: "white", castling: 0, enPassant: null, origin: null, board }]);
}

function singleTimelineGame(boards) {
  return {
    turn: "white",
    nextTimelineId: 1,
    nextBlackTimelineId: -1,
    checkedRoyals: [],
    royalCaptureBy: null,
    timelines: [{
      id: 0,
      row: 0,
      label: "T0",
      owner: "neutral",
      boards
    }]
  };
}

function startingBoard() {
  const board = emptyBoard();
  const backRank = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"];
  for (let x = 0; x < 8; x += 1) {
    board[0][x] = { color: "white", type: backRank[x] };
    board[1][x] = { color: "white", type: "pawn" };
    board[6][x] = { color: "black", type: "pawn" };
    board[7][x] = { color: "black", type: backRank[x] };
  }
  return board;
}

function emptyBoardWithKings() {
  const board = emptyBoard();
  board[0][4] = { color: "white", type: "king" };
  board[7][4] = { color: "black", type: "king" };
  return board;
}

function emptyBoard() {
  return Array.from({ length: 8 }, () => Array(8).fill(null));
}

main();
