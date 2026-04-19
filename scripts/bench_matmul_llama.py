"""
Сравнение X@W^T: W в bf16 (cuBLAS) vs W в int4 + Triton W4A16.
Размеры W — как у Llama-3.2-1B-Instruct; M ∈ {128, 512, 2048}.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

HIDDEN = 2048
INTERMEDIATE = 8192
NUM_KV_HEADS = 8
HEAD_DIM = 64
K_DIM = NUM_KV_HEADS * HEAD_DIM  # 512


def layer_specs():
    return [
        ("q_proj", HIDDEN, HIDDEN),
        ("k_proj", K_DIM, HIDDEN),
        ("gate_proj", INTERMEDIATE, HIDDEN),
        ("down_proj", HIDDEN, INTERMEDIATE),
    ]


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_fn(fn: Callable[[], object], warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        _ = fn()
    _sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    _sync()
    return (time.perf_counter() - t0) / iters


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


def _print_cuda_mismatch_hint():
    print(
        "\n--- Диагностика CUDA ---\n"
        "PyTorch сообщает, что CUDA недоступна.\n",
        file=sys.stderr,
    )
    smi = _nvidia_smi_cuda_version()
    if smi:
        print(f"  nvidia-smi: {smi!r}\n", file=sys.stderr)


def _should_run_sanity(
    *,
    sanity: bool,
    full_sanity: bool,
    layer_name: str,
    M: int,
    n_out: int,
    n_in: int,
) -> bool:
    if not sanity:
        return False
    if full_sanity:
        return M <= 512 and n_out <= 2048 and n_in <= 2048
    return layer_name == "q_proj" and M == 128 and n_out <= 2048 and n_in <= 2048


def run_benchmarks(
    *,
    warmup: int,
    iters: int,
    group_size: int,
    sanity: bool,
    full_sanity: bool,
    Ms: list[int],
    quant_backend: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    autotune_kernel: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from llmq.kernels.int4_quantize import quantize_fp16_to_int4
    from llmq.kernels.w4a16_matmul import w4a16_linear_bf16, w4a16_matmul_reference

    device = torch.device("cuda")
    meta = _collect_env_meta()
    rows: list[dict[str, Any]] = []

    kernel_config: dict[str, Any] = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "autotune": autotune_kernel,
    }

    def int4_mm(x_: torch.Tensor, qw_: Any) -> torch.Tensor:
        return w4a16_linear_bf16(
            x_,
            qw_,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
            autotune=autotune_kernel,
        )

    print("Llama-3.2-1B-Instruct: bf16 matmul vs int4 Triton W4A16")
    print(
        f"  group_size={group_size}  warmup={warmup}  iters={iters}  "
        f"Ms={Ms}  quant_backend={quant_backend}"
    )
    print(
        f"  kernel: BLOCK_M={block_m}  BLOCK_N={block_n}  BLOCK_K={block_k}  "
        f"num_warps={num_warps}  num_stages={num_stages}  autotune={autotune_kernel}"
    )
    print(
        f"  env: torch={meta['torch']}  cuda_available={meta['cuda_available']}",
        end="",
    )
    if meta.get("device_name"):
        print(f"  device={meta['device_name']}")
    else:
        print()

    with torch.inference_mode():
        for name, n_out, n_in in layer_specs():
            W_bf16 = torch.randn(n_out, n_in, device=device, dtype=torch.bfloat16) * 0.02

            tq0 = time.perf_counter()
            qw = quantize_fp16_to_int4(
                W_bf16,
                group_size=group_size,
                prefer_triton=(quant_backend == "triton"),
            )
            _sync()
            t_quant = time.perf_counter() - tq0

            n_el = n_out * n_in
            fp_bytes = n_el * 2
            q_bytes = qw.packed.numel() + qw.scales.numel() * 4
            ratio = fp_bytes / q_bytes

            print(
                f"\n=== {name}  W ({n_out}, {n_in})  "
                f"bf16_bytes≈{fp_bytes}  int4+scales≈{q_bytes}  ratio≈{ratio:.2f}x  "
                f"quant_time={t_quant:.3f}s  backend={quant_backend} ==="
            )

            for M in Ms:
                x = torch.randn(M, n_in, device=device, dtype=torch.bfloat16)

                _ = torch.matmul(x, W_bf16.T)
                _ = int4_mm(x, qw)
                _sync()

                if _should_run_sanity(
                    sanity=sanity,
                    full_sanity=full_sanity,
                    layer_name=name,
                    M=M,
                    n_out=n_out,
                    n_in=n_in,
                ):
                    t_ref0 = time.perf_counter()
                    y_ref = w4a16_matmul_reference(x, qw)
                    _sync()
                    t_ref = time.perf_counter() - t_ref0

                    y4 = int4_mm(x, qw)
                    diff = y_ref.to(torch.bfloat16).float() - y4.float()
                    err_mean = diff.abs().mean().item()
                    err_max = diff.abs().max().item()

                    print(
                        f"  M={M}: mean |ref-ker| = {err_mean:.6f}, "
                        f"max = {err_max:.6f} (sanity, ref_time={t_ref:.3f}s)"
                    )

                def run16():
                    return torch.matmul(x, W_bf16.T)

                def run4():
                    return int4_mm(x, qw)

                t16 = bench_fn(run16, warmup=warmup, iters=iters)
                t4 = bench_fn(run4, warmup=warmup, iters=iters)
                sp = t16 / t4 if t4 > 0 else float("inf")

                print(
                    f"  M={M:4d}  bf16: {t16 * 1000:.3f} ms   "
                    f"int4: {t4 * 1000:.3f} ms   speedup {sp:.2f}x"
                )

                rows.append(
                    {
                        "layer": name,
                        "N": n_out,
                        "K_in": n_in,
                        "M": M,
                        "quant_backend": quant_backend,
                        "kernel_config": dict(kernel_config),
                        "quant_time_ms": round(t_quant * 1000, 6),
                        "t_bf16_ms": round(t16 * 1000, 6),
                        "t_int4_ms": round(t4 * 1000, 6),
                        "speedup_bf16_over_int4": round(sp, 4),
                        "weight_bf16_bytes": fp_bytes,
                        "weight_int4_bytes": q_bytes,
                        "weight_compression_ratio": round(ratio, 4),
                    }
                )

    return rows, meta


def main():
    ap = argparse.ArgumentParser(description="Benchmark bf16 vs int4 matmul (Llama-1B shapes).")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--no-sanity", action="store_true")
    ap.add_argument("--full-sanity", action="store_true")
    ap.add_argument("--Ms", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--quant-backend", choices=["torch", "triton"], default="torch")

    kg = ap.add_argument_group("Triton W4A16 kernel (ignored when --autotune-kernel)")
    kg.add_argument("--block-m", type=int, default=64)
    kg.add_argument("--block-n", type=int, default=64)
    kg.add_argument("--block-k", type=int, default=128)
    kg.add_argument("--num-warps", type=int, default=4)
    kg.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--autotune-kernel", action="store_true", help="Let Triton pick BLOCK_* and warps/stages.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.fast:
        args.warmup = min(args.warmup, 2)
        args.iters = min(args.iters, 5)
        args.Ms = [m for m in args.Ms if m in (128, 512)] or [128, 512]

    try:
        import torch
    except ImportError:
        print("Установите зависимости: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        _print_cuda_mismatch_hint()
        print("CUDA недоступна — замеры на GPU невозможны.", file=sys.stderr)
        sys.exit(2)

    rows, meta = run_benchmarks(
        warmup=args.warmup,
        iters=args.iters,
        group_size=args.group_size,
        sanity=not args.no_sanity,
        full_sanity=args.full_sanity,
        Ms=args.Ms,
        quant_backend=args.quant_backend,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        autotune_kernel=args.autotune_kernel,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": meta, "rows": rows}
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.out}")

    print("\nГотово.")


if __name__ == "__main__":
    main()