"""Quality and speed benchmark for Llama-3.2-1B with W4A16 linear layers."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from llmq.modules.quant_linear import replace_linear_with_w4a16

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_time(fn: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0


def _load_wikitext2_text(split: str) -> str:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(row["text"] for row in ds if row["text"].strip())


def _token_blocks(
    tokenizer: Any,
    text: str,
    *,
    seq_len: int,
    max_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    token_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if max_tokens > 0:
        token_ids = token_ids[:max_tokens]
    usable = (token_ids.numel() // seq_len) * seq_len
    if usable < seq_len:
        raise ValueError(f"Not enough WikiText-2 tokens for seq_len={seq_len}.")
    return token_ids[:usable].view(-1, seq_len).to(device)


def _iter_batches(blocks: torch.Tensor, batch_size: int, max_batches: int) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    limit = blocks.shape[0] if max_batches <= 0 else min(blocks.shape[0], max_batches * batch_size)
    for start in range(0, limit, batch_size):
        batch = blocks[start : start + batch_size]
        if batch.shape[0] == batch_size:
            batches.append(batch)
    return batches


@torch.inference_mode()
def measure_perplexity(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    *,
    batch_size: int,
    max_batches: int,
) -> dict[str, float]:
    batches = _iter_batches(blocks, batch_size, max_batches)
    if not batches:
        raise ValueError("No batches available for perplexity measurement.")

    total_nll = 0.0
    total_tokens = 0
    for batch in batches:
        labels = batch.clone()
        out = model(input_ids=batch, labels=labels, use_cache=False)
        tokens = batch.shape[0] * max(batch.shape[1] - 1, 1)
        total_nll += out.loss.detach().float().item() * tokens
        total_tokens += tokens

    mean_nll = total_nll / total_tokens
    return {"nll": mean_nll, "perplexity": math.exp(mean_nll), "tokens": float(total_tokens)}


@torch.inference_mode()
def benchmark_batched_forward(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    *,
    warmup: int,
) -> dict[str, float]:
    if not batches:
        raise ValueError("No batches available for forward benchmark.")

    for batch in batches[:warmup]:
        _ = model(input_ids=batch, use_cache=False)
    _sync()

    def run_batches() -> None:
        for batch in batches:
            _ = model(input_ids=batch, use_cache=False)

    elapsed = _cuda_time(run_batches)
    tokens = sum(batch.numel() for batch in batches)
    return {
        "seconds": elapsed,
        "tokens": float(tokens),
        "tokens_per_second": tokens / elapsed,
        "batches": float(len(batches)),
    }


@torch.inference_mode()
def benchmark_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    batch_size: int,
    max_new_tokens: int,
    warmup: int,
    device: torch.device,
) -> dict[str, float | bool]:
    encoded = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True).to(device)
    if hasattr(model, "config"):
        model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    for _ in range(warmup):
        _ = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    _sync()

    outputs: torch.Tensor | None = None

    def run_generate() -> None:
        nonlocal outputs
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    elapsed = _cuda_time(run_generate)
    assert outputs is not None
    new_tokens = max(outputs.shape[1] - encoded.input_ids.shape[1], 0) * batch_size
    return {
        "seconds": elapsed,
        "new_tokens": float(new_tokens),
        "new_tokens_per_second": new_tokens / elapsed,
        "prompt_tokens": float(encoded.input_ids.numel()),
        "use_cache": True,
    }


def _load_model_and_tokenizer(args: argparse.Namespace) -> tuple[torch.nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.use_cache = True
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, tokenizer


def _variant_order(variants: str) -> list[str]:
    if variants == "both":
        return ["bf16", "w4a16"]
    return [variants]


def _run_variant(
    name: str,
    model: torch.nn.Module,
    tokenizer: Any,
    ppl_blocks: torch.Tensor,
    forward_blocks: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {"variant": name}

    result["perplexity"] = measure_perplexity(
        model,
        ppl_blocks,
        batch_size=args.ppl_batch_size,
        max_batches=args.ppl_batches,
    )
    print(f"  ppl: {result['perplexity']['perplexity']:.4f} (tokens={int(result['perplexity']['tokens'])})")

    forward_batches = _iter_batches(forward_blocks, args.forward_batch_size, args.forward_batches)
    result["forward"] = benchmark_batched_forward(model, forward_batches, warmup=args.warmup)
    print(f"  forward: {result['forward']['tokens_per_second']:.1f} tok/s ({result['forward']['seconds']:.3f}s)")

    result["generation"] = benchmark_generation(
        model,
        tokenizer,
        args.prompt,
        batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
        device=args.device,
    )
    print(
        f"  generation: {result['generation']['new_tokens_per_second']:.1f} new tok/s "
        f"({result['generation']['seconds']:.3f}s)"
    )

    return result


def create_llama_plots(payload: dict[str, Any], plots_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    results = payload["results"]
    variants = [item["variant"] for item in results]
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    perplexities = [item["perplexity"]["perplexity"] for item in results]
    ax.bar(variants, perplexities, color=["tab:blue", "tab:green"][: len(variants)])
    ax.set_title("WikiText-2 perplexity")
    ax.set_ylabel("Perplexity")
    ax.grid(axis="y", alpha=0.3)
    for idx, value in enumerate(perplexities):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    path = plots_dir / "llama_wikitext2_perplexity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = list(range(len(variants)))
    width = 0.34
    forward = [item["forward"]["tokens_per_second"] for item in results]
    generation = [item["generation"]["new_tokens_per_second"] for item in results]
    ax.bar([pos - width / 2 for pos in x], forward, width=width, label="Forward tok/s", color="tab:blue")
    ax.bar([pos + width / 2 for pos in x], generation, width=width, label="Generation new tok/s", color="tab:green")
    ax.set_title("Llama inference throughput")
    ax.set_ylabel("Tokens per second")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for idx, value in enumerate(forward):
        ax.text(idx - width / 2, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(generation):
        ax.text(idx + width / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = plots_dir / "llama_wikitext2_throughput.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    return outputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer = _load_model_and_tokenizer(args)
    text = _load_wikitext2_text(args.dataset_split)
    ppl_blocks = _token_blocks(tokenizer, text, seq_len=args.seq_len, max_tokens=args.ppl_tokens, device=args.device)
    forward_blocks = _token_blocks(
        tokenizer,
        text,
        seq_len=args.seq_len,
        max_tokens=args.forward_tokens,
        device=args.device,
    )

    payload: dict[str, Any] = {
        "model": args.model,
        "dataset": "wikitext-2-raw-v1",
        "dataset_split": args.dataset_split,
        "seq_len": args.seq_len,
        "device": str(args.device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else None,
        "results": [],
    }

    for variant in _variant_order(args.variants):
        if variant == "w4a16":
            skipped_linear = ["lm_head"]
            print(f"\nQuantizing Linear layers: group_size={args.group_size}, skip={skipped_linear}")
            t0 = time.perf_counter()
            stats = replace_linear_with_w4a16(
                model,
                group_size=args.group_size,
                prefer_triton_quant=True,
            )
            _sync()
            quant_seconds = time.perf_counter() - t0
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"  replaced={stats['replaced']} skipped={stats['skipped']} "
                f"params={stats['params_replaced']:,} quant_time={quant_seconds:.2f}s"
            )
            payload["quantization"] = {
                "group_size": args.group_size,
                "quant_backend": "triton",
                "skipped_linear": skipped_linear,
                "seconds": quant_seconds,
                **stats,
            }

        print(f"\n=== {variant} ===")
        payload["results"].append(_run_variant(variant, model, tokenizer, ppl_blocks, forward_blocks, args))

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Llama-3.2-1B-Instruct BF16 vs W4A16 on WikiText-2.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--variants", choices=["bf16", "w4a16", "both"], default="both")
    parser.add_argument("--group-size", type=int, default=128)

    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--ppl-tokens", type=int, default=32768)
    parser.add_argument("--ppl-batch-size", type=int, default=1)
    parser.add_argument("--ppl-batches", type=int, default=0, help="0 means all available PPL batches.")

    parser.add_argument("--forward-tokens", type=int, default=32768)
    parser.add_argument("--forward-batch-size", type=int, default=4)
    parser.add_argument("--forward-batches", type=int, default=16)

    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="The history of natural language processing begins")

    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)

    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    args.device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.device)
    return args


def main() -> None:
    args = parse_args()
    payload = run(args)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON: {args.out}")

    plots_dir = args.plots_dir
    if plots_dir is None and args.out is not None:
        plots_dir = args.out.parent / "plots"
    if plots_dir is not None:
        plots = create_llama_plots(payload, plots_dir)
        for path in plots:
            print(f"Plot: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
