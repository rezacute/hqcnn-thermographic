#!/usr/bin/env python3
"""
benchmark.py - inference latency and throughput (reviewer request).

    python benchmark.py --device cuda --out bench_t4.json
    python benchmark.py --device cpu  --out bench_cpu.json --threads 4

For the paper's inference table (Table VI), three variants, PQC executed by CUDA-Q:
    python benchmark.py --device cuda --variants classical,qi,pqc --qubits "" \
        --out results/bench_t4.json --tex results/tab_inference.tex

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
    """Per-image inference cost of the training kernel (cudaq_kernel.py) on CUDA-Q."""
    try:
        import cudaq
        from cudaq_kernel import hqcnn_kernel, readout_terms
    except Exception as exc:
        return dict(error=f"cudaq not available: {exc}")
    target = "nvidia" if device.type == "cuda" else "qpp-cpu"
    try:
        cudaq.set_target(target)
    except Exception as exc:
        return dict(error=f"cannot set target {target}: {exc}")

    out = dict(target=target)
    g = torch.Generator().manual_seed(0)
    for n in n_list:
        terms, ham = readout_terms(n)
        phi = ((torch.rand(n, generator=g) * 2 - 1) * math.pi / 4).tolist()
        theta = (torch.rand(2 * n * 2, generator=g) * 0.2 - 0.1).tolist()
        thf = (torch.rand(n, generator=g) * 0.2 - 0.1).tolist()
        run = lambda: cudaq.observe(hqcnn_kernel, ham, n, 2, 1, phi, theta, thf)
        res = timeit(run, torch.device("cpu"), 3, iters)
        out[str(n)] = dict(per_image_ms=res["median_ms"], p90_ms=res["p90_ms"])
        print(f"  cudaq {target} n={n:2d}: {res['median_ms']:.2f} ms per image (one observe call, {len(terms)} terms)")
    return out


def cudaq_training_step(n_list, device, train_images, batch=32, repeats=3):
    """Forward + parameter-shift backward of the PQC branch through CUDA-Q for one batch."""
    try:
        from pqc_cudaq import circuits_per_step
    except Exception as exc:
        return dict(error=str(exc))
    target = "nvidia" if device.type == "cuda" else "qpp-cpu"
    out = dict(target=target, batch=batch)
    steps = train_images // batch
    for n in n_list:
        br = PQCBranch(n_qubits=n, backend="cudaq", cudaq_target=target).to(device).train()
        h = torch.randn(batch, 512, device=device)
        times = []
        for _ in range(repeats + 1):
            br.zero_grad()
            t0 = time.perf_counter()
            br(h).pow(2).sum().backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        step = statistics.median(times[1:])
        out[str(n)] = dict(seconds_per_step=step, circuits_per_step=circuits_per_step(batch, n),
                           ms_per_circuit=1000 * step / circuits_per_step(batch, n),
                           minutes_per_epoch_branch=step * steps / 60)
        print(f"  cudaq training n={n:2d}: {step:.2f} s per batch of {batch} ({circuits_per_step(batch, n)} circuits, "
              f"{1000 * step / circuits_per_step(batch, n):.2f} ms each) -> about {step * steps / 60:.1f} min per epoch "
              f"for the circuit alone ({steps} steps)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--qubits", default="2,4,6,8,10,12,16")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    ap.add_argument("--pqc-backend", choices=["auto", "cudaq", "torch"], default="auto",
                    help="how the PQC variants execute their circuit (auto: CUDA-Q if installed)")
    ap.add_argument("--tex", default=None, help="LaTeX table path (default: --out with .tex)")
    ap.add_argument("--cudaq", action="store_true")
    ap.add_argument("--cudaq-qubits", default="4,8,12,16")
    ap.add_argument("--cudaq-train-qubits", default="8", help="qubit counts for the CUDA-Q training-step timing")
    ap.add_argument("--train-images", type=int, default=1090, help="training-set size, for the per-epoch estimate")
    ap.add_argument("--out", default="bench.json")
    a = ap.parse_args()
    device = torch.device(a.device)
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    info = dict(device=str(device), torch=torch.__version__, threads=torch.get_num_threads(),
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine(),
                warmup=a.warmup, iters=a.iters)
    backend = a.pqc_backend
    if backend == "auto":
        try:
            import cudaq  # noqa: F401
            backend = "cudaq"
        except Exception:
            backend = "torch"
    target = "nvidia" if device.type == "cuda" else "qpp-cpu"
    info.update(pqc_backend=backend, cudaq_target=target if backend == "cudaq" else None)
    print(f"benchmark on {info['device_name']} ({device}, {info['threads']} threads), "
          f"PQC circuit executed by {'CUDA-Q (' + target + ')' if backend == 'cudaq' else 'the PyTorch simulator'}")
    res = dict(info=info, models={}, branch_only={})

    for v in a.variants.split(","):
        model = HybridNet(v, pretrained=False, pqc_backend=backend if v in ("pqc", "pqc_noent") else "torch",
                          cudaq_target=target).to(device).eval()
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

    for n in [int(q) for q in a.qubits.split(",") if q.strip()]:
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
        res["cudaq_training"] = cudaq_training_step([int(q) for q in a.cudaq_train_qubits.split(",")], device,
                                                    a.train_images)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    names = dict(classical="Classical ResNet18", qi="HQ-CNN (QI)",
                 pqc="CUDA Quantum (8q)" if backend == "cudaq" else "PQC (8q)",
                 pqc_noent="PQC, no CNOT", mlp_twin="Circuit$\\to$MLP")
    mem = device.type == "cuda"
    rows = []
    for v, e in res["models"].items():
        cells = [names.get(v, v), f"{e['parameters']['total'] / 1e6:.2f}", f"{e['batch1']['median_ms']:.1f}",
                 f"{e['batch32']['images_per_s']:.0f}"]
        if mem:
            cells.append(f"{e['batch32']['peak_mem_mb']:.0f}")
        rows.append(" & ".join(cells) + r" \\")
    head = [r"\textbf{Model}", r"\textbf{Params (M)}", r"\textbf{ms/image}", r"\textbf{img/s}"]
    sub = ["", "", "(batch 1)", "(batch 32)"]
    if mem:
        head.append(r"\textbf{Memory (MB)}"); sub.append("(batch 32)")
    tex = "\n".join([r"\begin{tabular}{l" + "r" * (len(head) - 1) + "}", r"\toprule",
                     " & ".join(head) + r" \\", " & ".join(sub) + r" \\", r"\midrule", *rows,
                     r"\bottomrule", r"\end{tabular}"])
    tex_path = Path(a.tex) if a.tex else Path(a.out).with_suffix(".tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex + "\n")
    print(f"wrote {a.out} and {tex_path}")


if __name__ == "__main__":
    main()
