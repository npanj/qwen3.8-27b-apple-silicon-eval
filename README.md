# Qwen3.8-27B on Apple Silicon: Comprehensive Benchmarking & Quality Evaluation

Empirical benchmark suite, precision analysis, and quality evaluation comparing **Splash (Q4, Q8, Mixed, HQ)** against **MTPLX**, **MLX**, and **llama.cpp** running **Qwen3.8-27B** on Apple Silicon with Multi-Token Prediction (MTP).

---

## Executive Summary

* **Throughput (6.1x Peak Speedup)**: **Splash-Q4** hits **60.7 tok/s average** (peaking at **83.3 tok/s**), outperforming standard autoregressive MLX and llama.cpp baselines (9.9 tok/s) by over **6x**.
* **Direct 8-bit MTP Showdown**: Evaluating on the exact same 8-bit weight representation, **Splash-HQ** beats **MTPLX-Optimized-Quality** by **+39% overall** (36.9 vs 26.5 tok/s) and **+92% on mathematical reasoning** (54.8 vs 28.5 tok/s).
* **The "Reasoning Cliff"**: On competition-grade math (MATH-500 Problem 0: symbolic infinite double summation $\sum_{j=1}^\infty \sum_{k=1}^\infty \frac{1}{(j+k)^3} = p - q$), aggressive 4-bit and compressed 8-bit suffered numerical drift and failed. Upgrading sensitive layers (**Splash-Mixed** and **Splash-HQ**) resolved the cliff and **solved the problem correctly (`p - q`)**.
* **The Precision Speed Paradox**: Uncompressed native 8-bit (**Splash-HQ**) runs **faster** than compressed 8-bit (**Splash-Q8**) because higher-precision weights produce lower-entropy logits, increasing draft token acceptance rate and eliminating verification rollbacks.

---

## Grand Throughput Comparison (5 Standardized Domains)

Evaluated at `temperature=0.0`, `max_tokens=250` on identical hardware:

| Task / Domain | Prompt Description | Splash-Q4 | Splash-HQ (Native 8b) | Splash-Q8 (Compressed) | Splash-Mixed (Selective 8b) | MTPLX-Q8 (MTP D3) | MLX-Q8 (Stock AR) | llama.cpp (Q8_0 GGUF) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Math & Logic** | Bat & Ball algebraic derivation | **83.3 t/s** | **54.8 t/s** | 52.7 t/s | 48.6 t/s | 28.5 t/s | 9.9 t/s | 9.8 t/s |
| **Coding & Algorithms** | `merge_intervals` $O(N \log N)$ | **75.5 t/s** | **34.7 t/s** | 40.3 t/s | 35.0 t/s | 28.8 t/s | 9.9 t/s | 9.9 t/s |
| **Constraint Reasoning** | 3-chair spatial permutation | **59.3 t/s** | **39.3 t/s** | 37.4 t/s | 36.6 t/s | 27.7 t/s | 9.9 t/s | 9.9 t/s |
| **Domain Knowledge** | FlashAttn vs PagedAttn vs SpecDec | **39.0 t/s** | **21.9 t/s** | 22.8 t/s | 23.3 t/s | 23.8 t/s | 9.9 t/s | 10.0 t/s |
| **Nuanced Writing** | Memory bandwidth constraint (3 sentences) | **46.2 t/s** | **33.7 t/s** | 29.5 t/s | 27.0 t/s | 23.4 t/s | 9.9 t/s | 10.0 t/s |
| **AVERAGE SPEED** | **Across all 5 domains** | **60.7 tok/s** | **36.9 tok/s** | **36.5 tok/s** | **34.1 tok/s** | **26.5 tok/s** | **9.9 tok/s** | **9.9 tok/s** |
| **Speedup vs AR Baseline** | *(Baseline = 9.9 tok/s)* | **6.13x** | **3.73x** | **3.69x** | **3.44x** | **2.68x** | **1.00x** | **1.00x** |

---

## Quality & Reasoning Accuracy Scorecard

Evaluated on standard benchmarks with extended Chain-of-Thought thinking budgets (GPQA Diamond, AIME 2025, MATH-500, GSM8K):

| Benchmark | Focus | Splash-Q4 | Splash-Q8 | Splash-Mixed | Splash-HQ | MTPLX-Q8 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** | Grade School Arithmetic | **80.0%** (4/5) | **80.0%** (4/5) | — | — | **80.0%** (4/5) |
| **GPQA Diamond** | PhD-Level Physics & Science | 33.3% (1/3) | **66.7%** (2/3) | 33.3% (1/3) | 33.3% (1/3) | 20.0% (1/5) |
| **AIME 2025** | High School Math Olympiad | **66.7%** (2/3) | **66.7%** (2/3) | 33.3% (1/3) | **66.7%** (2/3) | 0.0% (0/5)* |
| **MATH-500** | Advanced Symbolic Math | 0.0% (0/3) | 0.0% (0/3) | **33.3%** (1/3) | **33.3%** (1/3) | — |
| **OVERALL ACCURACY** | **Cumulative Reasoning Score** | **33.3%** (3/9) | **44.4%** (4/9) | **33.3%** (3/9) | **44.4%** (4/9) | — |

*\*Note: Early MTPLX test was truncated before completion on AIME due to a 400-token ceiling before expanding generation limits.*

---

## Published Hugging Face Models

| Model | Hugging Face Repository | Precision Profile | Role |
| :--- | :--- | :--- | :--- |
| **Qwen3.8-27B-Splash-HQ** | [`nitinpanj/Qwen3.8-27B-Splash-HQ`](https://huggingface.co/nitinpanj/Qwen3.8-27B-Splash-HQ) | Native 8-bit uncompressed (64 layers + heads) | Maximum quality, zero quant error, high draft acceptance |
| **Qwen3.8-27B-Splash-Mixed** | [`nitinpanj/Qwen3.8-27B-Splash-Mixed`](https://huggingface.co/nitinpanj/Qwen3.8-27B-Splash-Mixed) | Top 8 sensitive layers (56–63) + heads in 8-bit | Memory-efficient hybrid solving the reasoning cliff |
| **Qwen3.8-27B-Splash-Q8** | [`nitinpanj/Qwen3.8-27B-Splash-Q8`](https://huggingface.co/nitinpanj/Qwen3.8-27B-Splash-Q8) | Compressed 8-bit baseline | High throughput 8-bit reference |

---

## Repository Structure

```
.
├── README.md               # Main benchmark overview and results
├── REPORT.md               # In-depth technical report and architectural analysis
├── scripts/
│   ├── run_all_models_standard_bench.py # 5-prompt grand comparison suite
│   ├── run_quality_benchmarks.py        # GPQA, AIME 2025, MATH-500 evaluator
│   ├── pack_mixed_q8.py                 # LRU-cached mixed-precision converter
│   ├── bench_mtplx_quality.py           # MTPLX MTP runner
│   ├── bench_mlx_q8.py                  # MLX stream generation benchmark
│   └── bench_llamacpp.py                # llama-cli Q8_0 GGUF benchmark
└── results/
    ├── splash_all_variants_5prompts.json
    ├── quality_benchmark_scorecard.json
    ├── mtplx_q8_results.json
    ├── mlx_q8_results.json
    ├── llamacpp_q8_results.json
    └── benchmark_results_q8_q4_mtp.json
```

---

## Quickstart & Reproduction

### 1. Run the 5-Prompt Head-to-Head Comparison
```bash
python3 scripts/run_all_models_standard_bench.py
```

### 2. Run the Quality Benchmark (GPQA, AIME, MATH-500)
```bash
python3 scripts/run_quality_benchmarks.py --limit 3 --max-tokens 800
```

### 3. Build a Selective Mixed-Precision Model
```bash
python3 scripts/pack_mixed_q8.py \
  --source-safetensors ~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Quality \
  --base-splash-dir install/models/incoai/Qwen3.8-27B-Splash-Q8 \
  --out-dir install/models/incoai/Qwen3.8-27B-Splash-Mixed \
  --upgrade-layers 56,57,58,59,60,61,62,63 \
  --upgrade-embedding \
  --upgrade-head
```

---

## License

Code and benchmark tools are released under the [Apache-2.0 License](LICENSE).
Model weights follow the original [Qwen License](https://github.com/QwenLM/Qwen2.5/blob/main/LICENSE).
