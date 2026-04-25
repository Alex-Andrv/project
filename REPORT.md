# Отчёт по проекту "Реализация triton кернелей для квантизации весов в LLM и инференса квантизованной модели"

## Цель

Цель проекта — реализовать и проверить `int4`-квантизацию весов линейных слоёв LLM и инференс формата **W4A16**: активации `bf16`, веса `int4`, fused dequantization + matmul на Triton.

В рамках текущей версии реализованы:

- Triton kernel для groupwise symmetric квантизации весов `bf16/fp16 -> int4` с упаковкой двух 4-битных значений в один `uint8`.
- Autotuned Triton kernel `W4A16`: `X_bf16 @ W_int4^T` с деквантизацией внутри matmul.
- Autotuned Triton baseline для обычного `bf16` matmul.
- `QuantLinearW4A16`, который заменяет `nn.Linear` и хранит веса в `int4 + scales`.
- Synthetic benchmark на формах линейных слоёв Llama-3.2-1B-Instruct.
- Benchmark полной модели Llama-3.2-1B-Instruct на WikiText-2: perplexity, batched forward throughput, generation throughput.
- CUDA-тесты корректности kernels и слоя.

## Окружение

Финальные замеры выполнялись на следующем setup-е:

| Параметр | Значение |
| --- | --- |
| GPU | NVIDIA A40 |
| VRAM | 46068 MiB |
| CUDA runtime | 12.8 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.6.0 |
| Модель | `unsloth/Llama-3.2-1B-Instruct` |
| Датасет качества | `wikitext-2-raw-v1`, split `test` |

## Финальные артефакты

- `results/matmul_llama.json` — основной synthetic matmul benchmark для `group_size=128`.
- `results/matmul_group_size_sweep.json` — sweep по `group_size ∈ {32, 64, 128, 256}`.
- `results/llama_wikitext2_w4a16.json` — full-model WikiText-2 benchmark для `group_size=128`.
- `results/llama_wikitext2_w4a16_gs64.json` — full-model WikiText-2 benchmark для `group_size=64`.
- `results/plots/` — графики основного matmul benchmark.
- `results/plots/group_size_sweep/` — графики sweep по `group_size`.
- `results/plots/llama_gs128/` и `results/plots/llama_gs64/` — графики full-model benchmark.

## Реализация

### Квантизация весов

Файл: `llmq/kernels/int4_quantize.py`.

Используется groupwise symmetric quantization по `K`-измерению матрицы весов `W` формы `(out_features, in_features)`. Для каждой группы размера `group_size=128` считается:

```text
scale = max(abs(w_group)) / 7
q = clamp(round(w / scale), -8, 7)
packed = q + 8
```

После смещения на `+8` два `int4` значения упаковываются в один `uint8`: младший nibble хранит чётный элемент, старший nibble хранит нечётный элемент.

Scales хранятся в `bf16`. Поэтому фактическое сжатие меньше ровно `4x`, так как кроме packed weights нужно хранить scales. Для `group_size=128` в экспериментах получилось примерно `3.88x`.

### W4A16 matmul

Файл: `llmq/kernels/w4a16_matmul.py`.

Kernel принимает:

- `X`: `(M, K)`, `bf16`
- packed `W`: `(N, K / 2)`, `uint8`
- `scales`: `(N, ceil(K / group_size))`, `bf16`

Внутри kernel:

1. Загружаются packed nibbles.
2. Восстанавливаются signed значения `[-8, 7]`.
3. Загружаются group scales.
4. Веса деквантизуются в `bf16`.
5. Выполняется `tl.dot` с accumulation в `fp32`.
6. Результат сохраняется в `bf16`.

Публичный API: `w4a16_linear_bf16(x, qw)`.

### QuantLinearW4A16

Файл: `llmq/modules/quant_linear.py`.

`QuantLinearW4A16` хранит:

- `packed`: packed `uint8` веса
- `scales`: group scales
- `bias`: dense bias, если он был в исходном `nn.Linear`

Также реализован helper `replace_linear_with_w4a16(...)`, который рекурсивно заменяет `nn.Linear` на `QuantLinearW4A16`. В benchmark полной модели `lm_head` не квантизуется.

## Synthetic matmul benchmark

Скрипт: `scripts/bench_matmul_llama.py`.

Команда финального запуска:

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

Формы матриц соответствуют линейным слоям Llama-3.2-1B-Instruct:

| Слой | W shape `(N, K)` | BF16 вес | INT4+scales | Сжатие | Quant time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `q_proj` | `(2048, 2048)` | 8.00 MiB | 2.06 MiB | 3.88x | 0.043 ms |
| `k_proj` | `(512, 2048)` | 2.00 MiB | 0.52 MiB | 3.88x | 0.040 ms |
| `v_proj` | `(512, 2048)` | 2.00 MiB | 0.52 MiB | 3.88x | 0.041 ms |
| `o_proj` | `(2048, 2048)` | 8.00 MiB | 2.06 MiB | 3.88x | 0.041 ms |
| `gate_proj` | `(8192, 2048)` | 32.00 MiB | 8.25 MiB | 3.88x | 0.145 ms |
| `up_proj` | `(8192, 2048)` | 32.00 MiB | 8.25 MiB | 3.88x | 0.156 ms |
| `down_proj` | `(2048, 8192)` | 32.00 MiB | 8.25 MiB | 3.88x | 0.156 ms |

![Weight storage by layer](./results/plots/matmul_weight_storage.png)

### Полные результаты matmul, ms

![Matmul latency M=128](./results/plots/matmul_latency_m128.png)

![Matmul latency M=512](./results/plots/matmul_latency_m512.png)

![Matmul latency M=2048](./results/plots/matmul_latency_m2048.png)

| Слой | M | torch bf16 | torch dequant + torch bf16 | Triton dequant + Triton bf16 | Triton W4A16 fused | Triton bf16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_proj` | 128 | 0.023 | 0.262 | 0.113 | 0.071 | 0.064 |
| `q_proj` | 512 | 0.054 | 0.294 | 0.115 | 0.211 | 0.065 |
| `q_proj` | 2048 | 0.171 | 0.415 | 0.223 | 0.619 | 0.195 |
| `k_proj` | 128 | 0.019 | 0.194 | 0.116 | 0.069 | 0.065 |
| `k_proj` | 512 | 0.018 | 0.242 | 0.114 | 0.070 | 0.064 |
| `k_proj` | 2048 | 0.046 | 0.183 | 0.114 | 0.212 | 0.063 |
| `v_proj` | 128 | 0.019 | 0.193 | 0.113 | 0.068 | 0.063 |
| `v_proj` | 512 | 0.019 | 0.185 | 0.112 | 0.069 | 0.062 |
| `v_proj` | 2048 | 0.046 | 0.182 | 0.113 | 0.211 | 0.062 |
| `o_proj` | 128 | 0.022 | 0.261 | 0.112 | 0.069 | 0.063 |
| `o_proj` | 512 | 0.055 | 0.294 | 0.113 | 0.210 | 0.063 |
| `o_proj` | 2048 | 0.171 | 0.414 | 0.223 | 0.632 | 0.197 |
| `gate_proj` | 128 | 0.078 | 1.081 | 0.168 | 0.210 | 0.076 |
| `gate_proj` | 512 | 0.185 | 1.189 | 0.283 | 0.609 | 0.195 |
| `gate_proj` | 2048 | 0.629 | 1.632 | 0.850 | 2.086 | 0.749 |
| `up_proj` | 128 | 0.078 | 1.083 | 0.170 | 0.212 | 0.075 |
| `up_proj` | 512 | 0.186 | 1.186 | 0.287 | 0.615 | 0.197 |
| `up_proj` | 2048 | 0.631 | 1.626 | 0.851 | 2.127 | 0.749 |
| `down_proj` | 128 | 0.076 | 1.076 | 0.205 | 0.267 | 0.075 |
| `down_proj` | 512 | 0.186 | 1.189 | 0.330 | 0.835 | 0.201 |
| `down_proj` | 2048 | 0.655 | 1.654 | 0.975 | 2.495 | 0.826 |

### Агрегированные отношения времени

Для первых четырёх колонок baseline - `torch_bf16_linear`. Две последние колонки отдельно сравнивают fused W4A16 с Triton `bf16` matmul и с вариантом `Triton dequantize -> Triton bf16 matmul`.

![Triton W4A16 slowdown](./results/plots/matmul_w4a16_slowdown.png)

![Triton W4A16 slowdown vs Triton bf16](./results/plots/matmul_w4a16_vs_triton_bf16_slowdown.png)

![Triton W4A16 slowdown vs separate Triton dequant + Triton bf16](./results/plots/matmul_w4a16_vs_triton_dequant_bf16_slowdown.png)

| M | torch dequant+torch bf16 / torch bf16 | Triton dequant+Triton bf16 / torch bf16 | Triton W4A16 / torch bf16 | Triton bf16 / torch bf16 | Triton W4A16 / Triton bf16 | Triton W4A16 / Triton dequant+Triton bf16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 12.30x | 4.19x | 3.22x | 2.20x | 1.93x | 0.89x |
| 512 | 7.58x | 3.04x | 3.76x | 1.76x | 2.74x | 1.68x |
| 2048 | 2.92x | 1.67x | 3.86x | 1.24x | 3.11x | 2.41x |

### Подбор group size

После основного benchmark был отдельно выполнен перебор по `group_size ∈ {32, 64, 128, 256}`:

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

![Group size tuning](./results/plots/group_size_sweep/matmul_group_size_tuning.png)

| Group size | Средний slowdown W4A16 / torch bf16 | Средний W4A16 / Triton dequant+Triton bf16 | Среднее сжатие |
| ---: | ---: | ---: | ---: |
| 32 | 4.42x | 2.00x | 3.56x |
| 64 | 3.61x | 1.63x | 3.76x |
| 128 | 3.60x | 1.66x | 3.88x |
| 256 | 3.58x | 1.67x | 3.94x |

По этим замерам `group_size=128` и `group_size=256` близки по скорости, но `256` даёт немного лучшее сжатие. Отдельная проверка качества ниже показывает, что для perplexity лучше использовать меньший `group_size=64`.

## Benchmark полной Llama модели на WikiText-2

Скрипт: `scripts/bench_llama_wikitext2.py`.

Команда финального запуска для `group_size=128`:

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

Для варианта с лучшим качеством был дополнительно выполнен финальный запуск с `group_size=64`:

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

Параметры:

- Модель: `unsloth/Llama-3.2-1B-Instruct`
- Dataset: `wikitext-2-raw-v1`, split `test`
- Sequence length: `1024`
- Perplexity: `32736` предсказываемых токенов
- Batched forward: `32768` токенов, batch size `4`, 8 batch-итераций
- Generation: batch size `1`, `128` новых токенов, KV-cache включён (`use_cache=True`)
- Квантизация: `group_size=128`, backend `triton`
- Заменено Linear слоёв: `112`
- Пропущено: `lm_head`
- Параметров в заменённых Linear слоях: `973,078,528`
- Время замены и квантизации: `0.560 s`

### Качество и скорость

![WikiText-2 perplexity](./results/plots/llama_gs128/llama_wikitext2_perplexity.png)

![Llama throughput](./results/plots/llama_gs128/llama_wikitext2_throughput.png)

| Вариант | Perplexity | NLL | Forward tok/s | Generation new tok/s |
| --- | ---: | ---: | ---: | ---: |
| `bf16` | 15.7707 | 2.7582 | 30420.2 | 69.2 |
| `w4a16` | 20.7829 | 3.0341 | 13069.0 | 43.8 |

Относительно `bf16`:

- Perplexity выросла с `15.77` до `20.78`, то есть примерно на `31.8%`.
- Batched forward стал примерно в `2.33x` медленнее.
- Generation с KV-cache стала примерно в `1.58x` медленнее.

### Дополнительная проверка качества group size

Чтобы проверить, не является ли просадка perplexity багом fused kernel, была сделана изоляционная проверка на первых `8192` токенах WikiText-2:

| Вариант | Perplexity |
| --- | ---: |
| `bf16` | 17.1186 |
| `w4a16 fused`, `group_size=128` | 21.8088 |
| `dense dequant reference`, `group_size=128` | 21.8200 |

`w4a16 fused` и dense dequant почти совпали. Значит просадка качества идёт от самой схемы квантизации, а не от ошибки Triton kernel или `QuantLinearW4A16`.

Также был выполнен быстрый перебор perplexity по `group_size` на тех же первых `8192` токенах:

| Group size | Perplexity |
| ---: | ---: |
| 64 | 20.4876 |
| 128 | 21.8088 |
| 256 | 23.5110 |

После этого полный WikiText-2 benchmark был повторён для `group_size=64`:

![WikiText-2 perplexity, group size 64](./results/plots/llama_gs64/llama_wikitext2_perplexity.png)

![Llama throughput, group size 64](./results/plots/llama_gs64/llama_wikitext2_throughput.png)

| Вариант | Group size | Perplexity | Forward tok/s | Generation new tok/s |
| --- | ---: | ---: | ---: | ---: |
| `bf16` | - | 15.7707 | 30476.4 | 63.2 |
| `w4a16` | 64 | 19.7086 | 12998.2 | 40.8 |
| `w4a16` | 128 | 20.7829 | 13069.0 | 43.8 |

Вывод: `group_size=64` заметно снижает просадку perplexity при почти той же скорости, но даёт меньшее сжатие весов (`3.76x` вместо `3.88x`). Для качества `group_size=64` выглядит предпочтительнее; для чуть лучшего memory compression можно оставить `128`.

## Тестирование

Добавлены CUDA-тесты в `tests/test_kernels_and_layers.py`.

Команда:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/test_kernels_and_layers.py -q
```

Что покрыто:

- `W4A16` fused kernel против reference `torch dequantize -> bf16 matmul` для всех типов Llama linear слоёв: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- `QuantLinearW4A16` против прямого вызова fused kernel.
- Triton `bf16` matmul против `torch.matmul`.
- Рекурсивная замена `nn.Linear` на `QuantLinearW4A16`, включая skip `lm_head`.

## Выводы

1. Реализация функционально работает: квантизация, упаковка, деквантизация и fused W4A16 matmul проходят sanity-checks и pytest.

2. Память весов уменьшается примерно в `3.88x` при `group_size=128`. Это близко к целевому `4x`, но ниже из-за хранения per-group scales.

3. Качество полной модели ухудшается заметно при `group_size=128`: perplexity на WikiText-2 выросла с `15.77` до `20.78`. Проверка dense dequant reference показала, что это не баг fused kernel, а эффект простой symmetric groupwise PTQ без calibration, zero-points, activation-aware методов или отдельной обработки чувствительных слоёв.

4. `group_size=64` выглядит более удачным quality/performance компромиссом: perplexity `19.71` вместо `20.78` у `group_size=128`, при почти той же скорости. Цена — меньшее сжатие весов, около `3.76x` вместо `3.88x`.

5. Текущий Triton W4A16 kernel пока не даёт ускорения на A40. Относительно `torch_bf16_linear` он в synthetic benchmark медленнее примерно в `3.2x-3.9x`, а относительно Triton `bf16` matmul — примерно в `1.9x-3.1x` в зависимости от `M`. В full-model benchmark это выражается в замедлении forward примерно в `2.33x` и generation с KV-cache примерно в `1.58x`.

6. Отдельный Triton `bf16` baseline близок к `torch` на больших M и больших слоях, но хуже на малых/узких формах. Сравнение W4A16 с Triton `bf16` показывает overhead упаковки, распаковки nibbles и применения scales внутри matmul, а сравнение с `Triton dequantize -> Triton bf16 matmul` отдельно проверяет, даёт ли fused kernel выигрыш относительно двух раздельных Triton kernels.

7. Честный baseline `Triton dequantize -> Triton bf16 matmul` показывает, что отдельная деквантизация на Triton намного быстрее torch-деквантизации, но текущий fused W4A16 не выигрывает стабильно. В среднем fused kernel немного быстрее только при `M=128` (`0.89x` latency ratio), а при `M=512` и `M=2048` медленнее примерно в `1.68x` и `2.41x`. Значит сама идея fusion остаётся важной для экономии памяти и удаления промежуточного dense weight tensor, но текущая fused реализация требует оптимизации, прежде чем её можно считать быстрее раздельного Triton dequant + Triton matmul.
