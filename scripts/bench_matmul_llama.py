"""Benchmark Llama-3.2-1B linear shapes across BF16 and W4A16 paths."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HIDDEN = 2048
INTERMEDIATE = 8192
NUM_KV_HEADS = 8
HEAD_DIM = 64
K_DIM = NUM_KV_HEADS * HEAD_DIM


def layer_specs() -> list[tuple[str, int, int]]:
    return [
        ("q_proj", HIDDEN, HIDDEN),
        ("k_proj", K_DIM, HIDDEN),
        ("v_proj", K_DIM, HIDDEN),
        ("o_proj", HIDDEN, HIDDEN),
        ("gate_proj", INTERMEDIATE, HIDDEN),
        ("up_proj", INTERMEDIATE, HIDDEN),
        ("down_proj", HIDDEN, INTERMEDIATE),
    ]


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_fn(fn: Callable[[], object], warmup: int = 5, iters: int = 20) -> float:
    import torch

    for _ in range(warmup):
        _ = fn()
    _sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters / 1000.0


def bench_quantize_weight(fn: Callable[[], Any], warmup: int = 5, iters: int = 20) -> tuple[Any, float]:
    import torch

    last = None
    for _ in range(warmup):
        last = fn()
    _sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        last = fn()
    end.record()
    end.synchronize()

    if last is None:
        last = fn()
        _sync()
    return last, start.elapsed_time(end) / iters / 1000.0


def _nvidia_smi_cuda_version() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _collect_env_meta() -> dict[str, Any]:
    import torch

    meta: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        meta["device_name"] = torch.cuda.get_device_name(0)
        meta["device_capability"] = torch.cuda.get_device_capability(0)
    smi = _nvidia_smi_cuda_version()
    if smi:
        meta["nvidia_smi_csv"] = smi
    return meta


def _print_cuda_mismatch_hint() -> None:
    print("\n--- CUDA diagnostics ---\nPyTorch reports that CUDA is unavailable.\n", file=sys.stderr)
    smi = _nvidia_smi_cuda_version()
    if smi:
        print(f"  nvidia-smi: {smi!r}\n", file=sys.stderr)


def _format_ms(seconds: float | None) -> str:
    if seconds is None:
        return "skipped"
    return f"{seconds * 1000:8.3f}"


def _print_result_table(entries: list[dict[str, Any]], baseline_ms: float | None) -> None:
    print("    backend                         ms      speedup_vs_torch")
    for entry in entries:
        time_s = entry.get("time_s")
        if time_s is None:
            print(f"    {entry['backend']:<28} skipped  {entry.get('skip_reason', '')}")
            continue
        speedup = baseline_ms / (time_s * 1000) if baseline_ms and time_s > 0 else float("nan")
        print(f"    {entry['backend']:<28} {_format_ms(time_s)}      {speedup:6.2f}x")


def create_matmul_plots(payload: dict[str, Any], plots_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    layers = [name for name, _, _ in layer_specs()]
    group_sizes = sorted({row.get("group_size", 128) for row in rows})
    primary_group_size = 128 if 128 in group_sizes else group_sizes[0]
    plot_rows = [row for row in rows if row.get("group_size", 128) == primary_group_size]
    ms_values = sorted({row["M"] for row in plot_rows})
    backend_labels = {
        "torch_bf16_linear": "torch bf16",
        "torch_dequant_bf16_matmul": "torch dequant + bf16",
        "triton_dequant_triton_bf16_matmul": "Triton dequant + Triton bf16",
        "triton_w4a16_fused": "Triton W4A16",
        "triton_bf16_matmul": "Triton bf16",
    }
    colors = {
        "torch_bf16_linear": "tab:blue",
        "torch_dequant_bf16_matmul": "tab:orange",
        "triton_dequant_triton_bf16_matmul": "tab:purple",
        "triton_w4a16_fused": "tab:green",
        "triton_bf16_matmul": "tab:red",
    }
    outputs: list[Path] = []

    for m_tokens in ms_values:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = list(range(len(layers)))
        width = min(0.8 / len(backend_labels), 0.18)
        for idx, backend in enumerate(backend_labels):
            values = [
                next(
                    row["time_ms"]
                    for row in plot_rows
                    if row["layer"] == layer and row["M"] == m_tokens and row["backend"] == backend
                )
                for layer in layers
            ]
            shift = (idx - (len(backend_labels) - 1) / 2) * width
            ax.bar(
                [pos + shift for pos in x],
                values,
                width=width,
                label=backend_labels[backend],
                color=colors[backend],
            )
        ax.set_title(f"Matmul latency, M={m_tokens}, group_size={primary_group_size}")
        ax.set_ylabel("Latency, ms")
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(ncols=2)
        fig.tight_layout()
        path = plots_dir / f"matmul_latency_m{m_tokens}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = list(range(len(layers)))
    width = 0.24
    for idx, m_tokens in enumerate(ms_values):
        values = []
        for layer in layers:
            bf16 = next(
                row["time_ms"]
                for row in plot_rows
                if row["layer"] == layer and row["M"] == m_tokens and row["backend"] == "torch_bf16_linear"
            )
            w4a16 = next(
                row["time_ms"]
                for row in plot_rows
                if row["layer"] == layer and row["M"] == m_tokens and row["backend"] == "triton_w4a16_fused"
            )
            values.append(w4a16 / bf16)
        shift = (idx - (len(ms_values) - 1) / 2) * width
        ax.bar([pos + shift for pos in x], values, width=width, label=f"M={m_tokens}")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_title(f"Triton W4A16 slowdown vs torch bf16, group_size={primary_group_size}")
    ax.set_ylabel("Latency ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "matmul_w4a16_slowdown.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = list(range(len(layers)))
    width = 0.24
    for idx, m_tokens in enumerate(ms_values):
        values = []
        for layer in layers:
            triton_bf16 = next(
                row["time_ms"]
                for row in plot_rows
                if row["layer"] == layer and row["M"] == m_tokens and row["backend"] == "triton_bf16_matmul"
            )
            w4a16 = next(
                row["time_ms"]
                for row in plot_rows
                if row["layer"] == layer and row["M"] == m_tokens and row["backend"] == "triton_w4a16_fused"
            )
            values.append(w4a16 / triton_bf16)
        shift = (idx - (len(ms_values) - 1) / 2) * width
        ax.bar([pos + shift for pos in x], values, width=width, label=f"M={m_tokens}")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_title(f"Triton W4A16 slowdown vs Triton bf16, group_size={primary_group_size}")
    ax.set_ylabel("Latency ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "matmul_w4a16_vs_triton_bf16_slowdown.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = list(range(len(layers)))
    width = 0.24
    for idx, m_tokens in enumerate(ms_values):
        values = []
        for layer in layers:
            dequant_triton = next(
                row["time_ms"]
                for row in plot_rows
                if row["layer"] == layer
                and row["M"] == m_tokens
                and row["backend"] == "triton_dequant_triton_bf16_matmul"
            )
            w4a16 = next(
                row["time_ms"]
                for row in plot_rows
                if row["layer"] == layer and row["M"] == m_tokens and row["backend"] == "triton_w4a16_fused"
            )
            values.append(w4a16 / dequant_triton)
        shift = (idx - (len(ms_values) - 1) / 2) * width
        ax.bar([pos + shift for pos in x], values, width=width, label=f"M={m_tokens}")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_title(f"Triton W4A16 vs separate Triton dequant + Triton bf16, group_size={primary_group_size}")
    ax.set_ylabel("Latency ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "matmul_w4a16_vs_triton_dequant_bf16_slowdown.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(layers)))
    bf16_mib = []
    int4_mib = []
    for layer in layers:
        row = next(
            row
            for row in plot_rows
            if row["layer"] == layer and row["M"] == ms_values[0] and row["backend"] == "torch_bf16_linear"
        )
        bf16_mib.append(row["weight_bf16_bytes"] / 2**20)
        int4_mib.append(row["weight_int4_bytes"] / 2**20)
    ax.bar([pos - 0.18 for pos in x], bf16_mib, width=0.36, label="bf16")
    ax.bar([pos + 0.18 for pos in x], int4_mib, width=0.36, label="int4 + scales")
    ax.set_title("Weight storage by layer")
    ax.set_ylabel("MiB")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "matmul_weight_storage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    if len(group_sizes) > 1:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        avg_slowdowns = []
        avg_compressions = []
        for group_size in group_sizes:
            slowdowns = []
            compressions = []
            for layer in layers:
                for m_tokens in ms_values:
                    bf16 = next(
                        row["time_ms"]
                        for row in rows
                        if row.get("group_size", 128) == group_size
                        and row["layer"] == layer
                        and row["M"] == m_tokens
                        and row["backend"] == "torch_bf16_linear"
                    )
                    w4a16 = next(
                        row["time_ms"]
                        for row in rows
                        if row.get("group_size", 128) == group_size
                        and row["layer"] == layer
                        and row["M"] == m_tokens
                        and row["backend"] == "triton_w4a16_fused"
                    )
                    slowdowns.append(w4a16 / bf16)
                compression = next(
                    row["weight_compression_ratio"]
                    for row in rows
                    if row.get("group_size", 128) == group_size
                    and row["layer"] == layer
                    and row["M"] == ms_values[0]
                    and row["backend"] == "torch_bf16_linear"
                )
                compressions.append(compression)
            avg_slowdowns.append(sum(slowdowns) / len(slowdowns))
            avg_compressions.append(sum(compressions) / len(compressions))
        labels = [str(group_size) for group_size in group_sizes]
        x = list(range(len(group_sizes)))
        ax.bar([pos - 0.18 for pos in x], avg_slowdowns, width=0.36, label="Avg W4A16 slowdown")
        ax.bar([pos + 0.18 for pos in x], avg_compressions, width=0.36, label="Avg compression")
        ax.axhline(1.0, color="black", linewidth=1)
        ax.set_title("Group size tuning summary")
        ax.set_ylabel("Ratio")
        ax.set_xlabel("group_size")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = plots_dir / "matmul_group_size_tuning.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)

    return outputs


def run_benchmarks(
    *,
    warmup: int,
    iters: int,
    group_size: int,
    Ms: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from llmq.kernels.bf16_matmul import bf16_linear_triton
    from llmq.kernels.int4_quantize import quantize_fp16_to_int4
    from llmq.kernels.w4a16_matmul import (
        dequantize_int4_weight_torch,
        dequantize_int4_weight_triton,
        w4a16_linear_bf16,
    )

    device = torch.device("cuda")
    meta = _collect_env_meta()
    rows: list[dict[str, Any]] = []

    print("Llama-3.2-1B-Instruct linear benchmark")
    print(f"  group_size={group_size}  warmup={warmup}  iters={iters}  Ms={Ms}  quant_backend=triton")
    print(f"  env: torch={meta['torch']}  cuda_available={meta['cuda_available']}", end="")
    print(f"  device={meta.get('device_name', 'unknown')}")

    with torch.inference_mode():
        warmup_w = torch.zeros((1, group_size), device=device, dtype=torch.bfloat16)
        _ = quantize_fp16_to_int4(warmup_w, group_size=group_size, prefer_triton=True)
        _sync()

        for name, n_out, n_in in layer_specs():
            w_bf16 = (torch.randn(n_out, n_in, device=device, dtype=torch.bfloat16) * 0.02).contiguous()

            qw, t_quant = bench_quantize_weight(
                lambda w_bf16=w_bf16: quantize_fp16_to_int4(
                    w_bf16,
                    group_size=group_size,
                    prefer_triton=True,
                ),
                warmup=warmup,
                iters=iters,
            )

            fp_bytes = w_bf16.numel() * w_bf16.element_size()
            q_bytes = qw.storage_bytes_weights_only()
            ratio = fp_bytes / q_bytes

            print(
                f"\n=== {name}  W=({n_out}, {n_in})  "
                f"bf16={fp_bytes / 2**20:.2f} MiB  int4+scales={q_bytes / 2**20:.2f} MiB  "
                f"ratio={ratio:.2f}x  quant={t_quant * 1000:.2f} ms ==="
            )

            for m_tokens in Ms:
                x = torch.randn(m_tokens, n_in, device=device, dtype=torch.bfloat16).contiguous()

                backends: list[tuple[str, Callable[[], torch.Tensor] | None, str | None]] = [
                    ("torch_bf16_linear", lambda x=x, w_bf16=w_bf16: torch.matmul(x, w_bf16.T), None),
                    (
                        "torch_dequant_bf16_matmul",
                        lambda x=x, qw=qw: torch.matmul(
                            x,
                            dequantize_int4_weight_torch(qw, dtype=torch.bfloat16).T,
                        ),
                        None,
                    ),
                    (
                        "triton_dequant_triton_bf16_matmul",
                        lambda x=x, qw=qw: bf16_linear_triton(
                            x,
                            dequantize_int4_weight_triton(qw),
                        ),
                        None,
                    ),
                    ("triton_w4a16_fused", lambda x=x, qw=qw: w4a16_linear_bf16(x, qw), None),
                    ("triton_bf16_matmul", lambda x=x, w_bf16=w_bf16: bf16_linear_triton(x, w_bf16), None),
                ]

                entries: list[dict[str, Any]] = []
                baseline_ms: float | None = None
                for backend, fn, skip_reason in backends:
                    if fn is None:
                        entry = {"backend": backend, "time_s": None, "skip_reason": skip_reason}
                    else:
                        try:
                            time_s = bench_fn(fn, warmup=warmup, iters=iters)
                        except Exception as exc:
                            entry = {"backend": backend, "time_s": None, "skip_reason": f"{type(exc).__name__}: {exc}"}
                        else:
                            entry = {"backend": backend, "time_s": time_s, "skip_reason": None}
                            if backend == "torch_bf16_linear":
                                baseline_ms = time_s * 1000
                    entries.append(entry)

                print(f"  M={m_tokens}")
                _print_result_table(entries, baseline_ms)

                for entry in entries:
                    rows.append(
                        {
                            "layer": name,
                            "N": n_out,
                            "K_in": n_in,
                            "M": m_tokens,
                            "backend": entry["backend"],
                            "group_size": group_size,
                            "time_ms": None if entry["time_s"] is None else round(entry["time_s"] * 1000, 6),
                            "skip_reason": entry.get("skip_reason"),
                            "quant_backend": "triton",
                            "quant_time_ms": round(t_quant * 1000, 6),
                            "weight_bf16_bytes": fp_bytes,
                            "weight_int4_bytes": q_bytes,
                            "weight_compression_ratio": round(ratio, 4),
                        }
                    )

    return rows, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark BF16 and W4A16 matmul on Llama-1B linear shapes.")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--group-sizes", type=int, nargs="+", default=None)
    ap.add_argument("--Ms", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--plots-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    try:
        import torch
    except ImportError:
        print('Install dependencies with: pip install -e ".[dev]"', file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        _print_cuda_mismatch_hint()
        print("CUDA is unavailable; GPU benchmarks cannot run.", file=sys.stderr)
        sys.exit(2)

    group_sizes = args.group_sizes or [args.group_size]
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] | None = None
    for group_size in group_sizes:
        group_rows, group_meta = run_benchmarks(
            warmup=args.warmup,
            iters=args.iters,
            group_size=group_size,
            Ms=args.Ms,
        )
        rows.extend(group_rows)
        meta = group_meta
    assert meta is not None

    payload = {"meta": meta, "group_sizes": group_sizes, "rows": rows}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.out}")

    plots_dir = args.plots_dir
    if plots_dir is None and args.out is not None:
        plots_dir = args.out.parent / "plots"
    if plots_dir is not None:
        plots = create_matmul_plots(payload, plots_dir)
        for path in plots:
            print(f"Plot: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
