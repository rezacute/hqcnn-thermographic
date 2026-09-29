#!/usr/bin/env python3
"""
benchmark.py - inference latency and throughput (reviewer request).

    python benchmark.py --device cuda --out bench_t4.json
    python benchmark.py --device cpu  --out bench_cpu.json --threads 4

Timing does not depend on trained weights, so models are built with random weights.
For every variant: latency for one image (batch 1) and throughput at batch 32, median and
90th percentile over --iters timed runs after --warmup untimed runs (CUDA synchronised).
Also times the quantum branch alone for n = 2..16 qubits (its share of the cost), and,
with --cudaq, the same circuit executed by CUDA-Q (one cudaq.observe call per image).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch

from models import VARIANTS, HybridNet
from pqc_torch import PQCBranch


def timeit(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - t0))
    times.sort()
    return dict(median_ms=statistics.median(times), p90_ms=times[int(0.9 * (len(times) - 1))], iters=iters)


def cudaq_latency(n_list, iters, device):
    try:
        import cudaq
        from cudaq import spin
    except Exception as exc:
        return dict(error=f"cudaq not available: {exc}")
    target = "nvidia" if device.type == "cuda" else "qpp-cpu"
    try:
        cudaq.set_target(target)
    except Exception as exc:
        return dict(error=f"cannot set target {target}: {exc}")

    @cudaq.kernel
    def hqcnn_kernel(n: int, layers: int, phi: list[float], theta: list[float], theta_final: list[float]):
        q = cudaq.qvector(n)
        for k in range(n):
            ry(phi[k], q[k])
        for layer in range(layers):
            for k in range(n):
                ry(theta[(layer * n + k) * 2], q[k])
                rz(theta[(layer * n + k) * 2 + 1], q[k])
            for k in range(n - 1):
                x.ctrl(q[k], q[k + 1])
            x.ctrl(q[n - 1], q[0])
        for k in range(n):
            ry(theta_final[k], q[k])

    out = dict(target=target)
    g = torch.Generator().manual_seed(0)
    for n in n_list:
        terms = [spin.z(k) for k in range(n)] + [spin.z(k) * spin.z(k + 1) for k in range(n - 1)]
        ham = terms[0]
        for t in terms[1:]:
            ham = ham + t
        phi = ((torch.rand(n, generator=g) * 2 - 1) * math.pi / 4).tolist()
        theta = (torch.rand(2 * n * 2, generator=g) * 0.2 - 0.1).tolist()
        thf = (torch.rand(n, generator=g) * 0.2 - 0.1).tolist()
        run = lambda: cudaq.observe(hqcnn_kernel, ham, n, 2, phi, theta, thf)
        res = timeit(run, torch.device("cpu"), 3, iters)
        out[str(n)] = dict(per_image_ms=res["median_ms"], p90_ms=res["p90_ms"])
        print(f"  cudaq {target} n={n:2d}: {res['median_ms']:.2f} ms per image (one observe call, {len(terms)} terms)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--qubits", default="2,4,6,8,10,12,16")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    ap.add_argument("--cudaq", action="store_true")
    ap.add_argument("--cudaq-qubits", default="4,8,12,16")
    ap.add_argument("--out", default="bench.json")
    a = ap.parse_args()
    device = torch.device(a.device)
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    info = dict(device=str(device), torch=torch.__version__, threads=torch.get_num_threads(),
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine(),
                warmup=a.warmup, iters=a.iters)
    print(f"benchmark on {info['device_name']} ({device}, {info['threads']} threads)")
    res = dict(info=info, models={}, branch_only={})

    for v in a.variants.split(","):
        model = HybridNet(v, pretrained=False).to(device).eval()
        entry = dict(parameters=model.count_parameters())
        for bs in (1, 32):
            x = torch.randn(bs, 3, 224, 224, device=device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                t = timeit(lambda: model(x), device, a.warmup, a.iters)
            t["per_image_ms"] = t["median_ms"] / bs
            t["images_per_s"] = 1000 * bs / t["median_ms"]
            if device.type == "cuda":
                t["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
            entry[f"batch{bs}"] = t
        res["models"][v] = entry
        print(f"  {v:10s} batch 1: {entry['batch1']['median_ms']:.2f} ms   "
              f"batch 32: {entry['batch32']['images_per_s']:.0f} img/s")

    for n in [int(q) for q in a.qubits.split(",")]:
        br = PQCBranch(n_qubits=n).to(device).eval()
        entry = {}
        for bs in (1, 32):
            h = torch.randn(bs, 512, device=device)
            with torch.inference_mode():
                t = timeit(lambda: br(h), device, a.warmup, a.iters)
            t["per_image_ms"] = t["median_ms"] / bs
            entry[f"batch{bs}"] = t
        res["branch_only"][str(n)] = entry
        print(f"  PQC branch n={n:2d}: batch 1 {entry['batch1']['median_ms']:.3f} ms, "
              f"batch 32 {entry['batch32']['per_image_ms']:.3f} ms/image")

    if a.cudaq:
        res["cudaq"] = cudaq_latency([int(q) for q in a.cudaq_qubits.split(",")], max(5, a.iters // 5), device)

    Path(a.out).write_text(json.dumps(res, indent=2))
    names = dict(classical="Classical", qi="QI", pqc="PQC (8q)", pqc_noent="PQC no CNOT", mlp_twin="MLP twin")
    rows = [f"{names.get(v, v)} & {e['parameters']['total'] / 1e6:.2f}M & {e['batch1']['median_ms']:.2f} & "
            f"{e['batch32']['images_per_s']:.0f} \\\\" for v, e in res["models"].items()]
    tex = "\n".join([r"\begin{tabular}{lrrr}", r"\toprule",
                     r"Variant & Params & ms/image (bs 1) & img/s (bs 32) \\", r"\midrule", *rows,
                     r"\bottomrule", r"\end{tabular}"])
    Path(a.out).with_suffix(".tex").write_text(tex)
    print(f"wrote {a.out} and {Path(a.out).with_suffix('.tex')}")


if __name__ == "__main__":
    main()
