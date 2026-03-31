# llmq-triton

Учебно‑исследовательский проект про int4‑квантизацию весов линейных слоёв и матмул **W4A16** (активации **bf16**, веса **int4**) на Triton. Размерности бенчмарка подобраны под **Llama‑3.2‑1B‑Instruct** (см. [config](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/raw/main/config.json)).

## Требования

- Linux + NVIDIA GPU
- Python 3.10+
- **Важно:** сборка PyTorch должна соответствовать драйверу/CUDA. Если `torch.cuda.is_available()` = `False`, чаще всего помогает поставить PyTorch под подходящую CUDA (см. `requirements-gpu.txt`).

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -U pip numpy triton
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Остальные зависимости: `pip install -r requirements.txt`

## Запуск бенчмарка

```bash
pip install -r requirements.txt
python scripts/bench_matmul_llama.py
```

Подробный отчёт по первым экспериментам и плану проекта: [REPORT.md](REPORT.md).

## Структура

- `llmq/kernels/int4_quantize.py` — построчная квантизация с группировкой по `K`, упаковка 2×int4 в байт
- `llmq/kernels/w4a16_matmul.py` — matmul `w4a16_linear_bf16` и референс для сверки
- `scripts/bench_matmul_llama.py` — сравнение `X @ W_bf16^T` и `X @ W_int4^T` для `M ∈ {128, 512, 2048}`

## Участники

- Александр Андреев
- Михаил Маслов
- Артём Матвеев

## Лицензия

Код предоставляется “как есть” для учебного/исследовательского проекта. При необходимости добавьте файл `LICENSE`.
# project
