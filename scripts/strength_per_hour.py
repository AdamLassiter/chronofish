"""Compare frozen and experimental trainer binaries at equal wall-clock budgets."""
import argparse
import csv
import json
import re
import random
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path


def train(binary, directory, args, seed, persistent):
    command = [binary, "--work-dir", directory, "--seed", str(seed), "--iterations",
               "1000000", "--training-device", args.device, "--model-size", args.model_size,
               "--threads", str(args.threads)]
    if persistent:
        command.append("--persistent")
    command.extend(args.trainer_args)
    with (directory / "training.log").open("w") as log:
        start = time.monotonic()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=args.seconds)
            raise RuntimeError(f"Trainer exited early. See {log.name}")
        except subprocess.TimeoutExpired:
            process.kill()
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    return time.monotonic() - start


def evaluate(benchmark, candidate, baseline, output):
    with output.open("w") as stream:
        subprocess.run([benchmark, "--match", candidate, baseline], stdout=stream, check=True)
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["ruleset"], row["opening"]].append(
            {"win": 1, "loss": 0, "terminal_draw": 0.5, "action_limit": 0.5}[row["outcome"]])
    scores = [statistics.mean(pair) for pair in pairs.values()]
    rng = random.Random(401)
    samples = sorted(statistics.mean(rng.choices(scores, k=len(scores))) for _ in range(4000))
    return dict(outcomes=dict(Counter(row["outcome"] for row in rows)),
                score=statistics.mean(scores), opening_pair_ci95=[samples[100], samples[3899]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "trainer", "benchmark", "snapshot", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "vulkan"], required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--model-size", default="balanced")
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.trainer_args[:1] == ["--"]:
        args.trainer_args.pop(0)
    if args.output.resolve().is_relative_to(args.snapshot.resolve()):
        parser.error("choose an output directory outside the starting snapshot")
    if args.seconds <= 0 or args.threads <= 0:
        parser.error("seconds and threads must be positive")
    for filename in ("candidate-v2.json", "best-v2.json", "replay-v2.bin.zst"):
        if not (args.snapshot / filename).is_file():
            parser.error(f"missing snapshot file {filename}")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "initial"
    shutil.copytree(args.snapshot, snapshot)
    for seed in (101, 211, 307):
        for label, binary in (("baseline", args.baseline), ("optimized", args.trainer),
                              ("adam", args.trainer)):
            work = args.output / f"{seed}-{label}"
            shutil.copytree(snapshot, work)
            elapsed = train(binary.resolve(), work, args, seed, label == "adam")
            candidate = work / "candidate-v2.json"
            steps = json.loads(candidate.read_bytes())["training_steps"]
            initial_steps = json.loads((snapshot / "candidate-v2.json").read_bytes())["training_steps"]
            log = (work / "training.log").read_text()
            positions = sum(int(n.replace(",", "")) for n in re.findall(
                r"game \d+ · [^\n]+? · ([\d,]+) positions? ·", log))
            match = evaluate(args.benchmark.resolve(), candidate, snapshot / "best-v2.json",
                             work / "held-out.csv")
            report = dict(seed=seed, arm=label, elapsed_seconds=elapsed, training_steps=steps,
                          optimizer_updates=steps-initial_steps, generated_positions=positions,
                          device=args.device, **match)
            (work / "result.json").write_text(json.dumps(report, indent=2))
            print(report, flush=True)


if __name__ == "__main__":
    main()
