# Реализация triton кернелей для квантизации весов в LLM и инференса квантизованной модели

Учебно-исследовательский проект про int4-квантизацию весов линейных слоёв LLM и инференс **W4A16**: активации `bf16`, веса `int4`, fused dequantization + matmul на Triton.

Основная целевая модель для замеров: [unsloth/Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct). Размерности synthetic matmul benchmark взяты из линейных слоёв этой модели.

Зависимости проекта описаны в `pyproject.toml`.

## Установка

Создайте виртуальное окружение и установите проект с dev-зависимостями:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Тесты

Основной набор тестов запускается через `pytest`:

```bash
.venv/bin/python -m pytest
```

## Результаты

Финальные JSON и графики лежат в `results/`:

- `results/matmul_llama.json` — основной synthetic matmul benchmark для `group_size=128`
- `results/matmul_group_size_sweep.json` — перебор по `group_size ∈ {32, 64, 128, 256}`
- `results/llama_wikitext2_w4a16.json` — full-model benchmark для `group_size=128`
- `results/llama_wikitext2_w4a16_gs64.json` — full-model benchmark для `group_size=64`
- `results/plots/` — PNG-графики для отчёта и презентации

Короткий вывод по качеству: `group_size=64` даёт лучшее perplexity (`19.71` против `20.78` у `group_size=128`) почти без изменения скорости, но с чуть меньшим сжатием весов (`3.76x` против `3.88x`). Для качества лучше использовать `64`, для чуть лучшего сжатия — `128`.

## Matmul Benchmark

Скрипт `scripts/bench_matmul_llama.py` сравнивает матричное умножение на формах линейных слоёв Llama-3.2-1B-Instruct для `M ∈ {128, 512, 2048}`.

Основной запуск из отчёта:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench_matmul_llama.py \
  --gpu 0 \
  --Ms 128 512 2048 \
  --warmup 10 \
  --iters 50 \
  --group-size 128 \
  --out results/matmul_llama.json \
  --plots-dir results/plots
```

В benchmark выводятся строки:

- `torch_bf16_linear` — обычный `torch.matmul` для dense `bf16` весов
- `torch_dequant_bf16_matmul` — dequantization int4 весов в torch, затем `bf16` matmul
- `triton_dequant_triton_bf16_matmul` — dequantization int4 весов отдельным Triton kernel, затем Triton `bf16` matmul
- `triton_w4a16_fused` — fused Triton kernel: dequantization + matmul
- `triton_bf16_matmul` — отдельный Triton `bf16` matmul baseline

Основной вывод по fusion: относительно отдельного `Triton dequantize -> Triton bf16 matmul` fused W4A16 немного быстрее только при `M=128` (`0.89x` latency ratio), но проигрывает при `M=512` (`1.68x`) и `M=2048` (`2.41x`). Fusion всё ещё полезен как направление, потому что убирает промежуточный dense weight tensor, но текущий fused kernel требует оптимизации.

Подбор `group_size`:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench_matmul_llama.py \
  --gpu 0 \
  --Ms 128 512 2048 \
  --warmup 3 \
  --iters 10 \
  --group-sizes 32 64 128 256 \
  --out results/matmul_group_size_sweep.json \
  --plots-dir results/plots/group_size_sweep
```

## Llama WikiText-2 Benchmark

Скрипт `scripts/bench_llama_wikitext2.py` запускает уже полную модель и может сравнить baseline `bf16` с моделью, где `nn.Linear` заменены на `QuantLinearW4A16`.

Он измеряет:

- perplexity на `wikitext-2-raw-v1`
- скорость батчевого forward inference на токенах WikiText-2
- скорость генерации через `model.generate`

Полный запуск из отчёта для `group_size=128`:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench_llama_wikitext2.py \
  --gpu 0 \
  --variants both \
  --seq-len 1024 \
  --ppl-tokens 32768 \
  --forward-batch-size 4 \
  --forward-batches 16 \
  --max-new-tokens 128 \
  --out results/llama_wikitext2_w4a16.json \
  --plots-dir results/plots/llama_gs128
```

Повторный полный запуск из отчёта для `group_size=64`:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench_llama_wikitext2.py \
  --gpu 0 \
  --variants both \
  --group-size 64 \
  --seq-len 1024 \
  --ppl-tokens 32768 \
  --forward-batch-size 4 \
  --forward-batches 16 \
  --max-new-tokens 128 \
  --out results/llama_wikitext2_w4a16_gs64.json \
  --plots-dir results/plots/llama_gs64
```

`lm_head` не квантизуется: benchmark заменяет только внутренние `nn.Linear` слои модели.

Первый запуск может скачать модель и датасет из Hugging Face.

## Структура

- `llmq/kernels/int4_quantize.py` — groupwise symmetric `bf16/fp16 -> int4`, упаковка 2 значения в `uint8`
- `llmq/kernels/w4a16_matmul.py` — autotuned Triton W4A16 fused dequantization + matmul
- `llmq/kernels/bf16_matmul.py` — autotuned Triton `bf16` matmul baseline
- `llmq/modules/quant_linear.py` — `QuantLinearW4A16` и helper для замены `nn.Linear`
- `scripts/bench_matmul_llama.py` — synthetic matmul benchmark на Llama-shaped матрицах
- `scripts/bench_llama_wikitext2.py` — quality и inference benchmark полной Llama модели

## Участники

- Александр Андреев
- Михаил Маслов
- Артём Матвеев
