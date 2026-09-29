#!/usr/bin/env python3
"""
run_grid.py - run every configuration the revised paper needs, spread over the GPUs.

    python run_grid.py --data DATA --grid full --seeds 0,1,2,3,4 --gpus 0,1,2,3

Grids
  core    the main comparison at n = 8:
            classical | qi | pqc | pqc_noent (no CNOTs) | mlp_twin (circuit -> classical map)
  sweep   pqc at n = 2, 4, 6, 10, 12, 16  (n = 8 comes from core)
  fixes   the paper's "training fixes" re-tested on a simulated circuit (n = 8):
            pi encoding + BN | pi/4 without BN | pi without BN  (pi/4 + BN comes from core)
  full    core + sweep + fixes      (14 configurations)
  quick   core only                 (use with --seeds 0,1,2 for a first look)

Finished runs (metrics.json present) are skipped, so the command can be re-run after an
interruption. One job per GPU at a time; logs go to RUNS/_logs/.
"""
from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

CORE = [dict(variant="classical"), dict(variant="qi"),
        dict(variant="pqc"), dict(variant="pqc_noent"), dict(variant="mlp_twin")]
SWEEP = [dict(variant="pqc", n_qubits=n) for n in (2, 4, 6, 10, 12, 16)]
FIXES = [dict(variant="pqc", enc_scale="pi"), dict(variant="pqc", no_bn=True),
         dict(variant="pqc", enc_scale="pi", no_bn=True)]
GRIDS = dict(core=CORE, quick=CORE, sweep=SWEEP, fixes=FIXES, full=CORE + SWEEP + FIXES)


def to_args(cfg: dict) -> list[str]:
    out = []
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            out.append(flag)
        elif v is not False:
            out += [flag, str(v)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--grid", choices=list(GRIDS), default="full")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids, or 'cpu'")
    ap.add_argument("--extra", default="", help="extra arguments passed to train.py, quoted")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    jobs = [(cfg, int(s)) for s in a.seeds.split(",") for cfg in GRIDS[a.grid]]
    here = Path(__file__).resolve().parent
    logs = Path(a.runs) / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    print(f"{len(jobs)} runs ({len(GRIDS[a.grid])} configurations x {len(a.seeds.split(','))} seeds)")
    failures = []

    def worker(gpu: str):
        while True:
            try:
                cfg, seed = q.get_nowait()
            except queue.Empty:
                return
            cmd = [sys.executable, str(here / "train.py"), "--data", a.data, "--runs", a.runs,
                   "--seed", str(seed)] + to_args(cfg) + (a.extra.split() if a.extra else [])
            env = dict(**__import__("os").environ)
            if gpu == "cpu":
                cmd += ["--device", "cpu"]
            else:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            tag = "_".join(f"{k}{v}" for k, v in cfg.items()).replace("/", "") + f"_seed{seed}"
            if a.dry_run:
                print("DRY", gpu, " ".join(cmd)); continue
            t0 = time.time()
            with open(logs / f"{tag}.log", "w") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            status = "ok" if rc == 0 else f"FAILED (exit {rc})"
            print(f"[gpu {gpu}] {tag}: {status} in {(time.time() - t0) / 60:.1f} min", flush=True)
            if rc != 0:
                failures.append(tag)

    threads = [threading.Thread(target=worker, args=(g.strip(),)) for g in a.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        print(f"\n{len(failures)} run(s) failed - see {logs}/: {failures}")
        sys.exit(1)
    print("\nall runs finished")


if __name__ == "__main__":
    main()
