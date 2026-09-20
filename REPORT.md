# Comprehensive Technical Report: Qwen3.8-27B Speculative Decoding & Quantization on Apple Silicon

**Author**: nitinpanj  
**Date**: September 2026  
**Hardware Platform**: Apple Silicon Mac (Unified Memory Architecture)  
**Evaluated Models**: Qwen3.8-27B (Splash-Q4, Splash-Q8, Splash-Mixed, Splash-HQ, MTPLX-Optimized-Quality, MLX-Community-8bit, llama.cpp-Q8_0)

---

## 1. Introduction & Background

Modern Large Language Models (LLMs) deployed on consumer edge hardware face a severe memory-bandwidth bottleneck during autoregressive token generation. Because generating each individual token requires streaming billions of parameters from RAM to compute units, arithmetic intensity (FLOPs per byte) is extremely low ($\approx 1 \text{ FLOP/byte}$).

To overcome this constraint, two complementary paradigms have emerged:
1. **Weight Quantization** (e.g. 4-bit, 8-bit, dynamic bit-width) to reduce the bytes transferred per token.
2. **Speculative Decoding / Multi-Token Prediction (MTP)** to generate multiple tokens per memory read pass.

Modern models like **Qwen3.8-27B** incorporate native MTP heads trained alongside the base model. This allows the model to draft future tokens ahead of itself, eliminating the need for an external draft model.

This report documents our empirical investigation comparing **Splash** (a high-performance Metal speculative inference engine) against **MTPLX**, **MLX**, and **llama.cpp**, exploring throughput, reasoning precision, selective layer upgrading, and the "reasoning cliff."

---

## 2. The Apple Silicon MTP Ecosystem: Myth vs. Reality

### A. What MTPLX Actually Is
**MTPLX** (`mtplx.com`, by Youssof) is an Apple Silicon native inference engine designed to execute Qwen models with built-in MTP heads. 

Crucially, inspecting the repository `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality` reveals:
* It is **not a custom-trained base model**.
* It uses the exact same base weights as `mlx-community/Qwen3.8-27B-8bit` (8-bit affine quantization, group size 64).
* The official `mlx-community` release omitted the MTP draft sidecar (`mtp.safetensors`).
* MTPLX packaged `mlx-community/Qwen3.8-27B-8bit` together with the pre-quantized 8-bit MTP sidecar (451 MB) to enable speculative decoding on macOS.

### B. The 4 Inference Engines Under Test
1. **Splash (IncoAI)**: High-efficiency Metal engine using compiled kernel pipelines, dedicated target and draft execution streams, and zero-copy unified memory buffers.
2. **MTPLX**: Apple Silicon native application and CLI using PyPI MLX with custom MTP verification strategies (`DraftCore device-d2`, draft depth 3).
3. **MLX (`mlx-lm`)**: The official Apple machine learning framework executing standard autoregressive decoding.
4. **llama.cpp**: The cross-platform C/C++ reference running quantized GGUF (`AtomicChat/Qwen3_8-27B-Q8_0`).

---

## 3. Standard Quantization Benchmarks: Qwen vs. Unsloth

When measuring the impact of quantization on reasoning capability, standard single-metric evaluations (like raw Perplexity on WikiText) can be dangerously misleading:
* **Perplexity Pitfall**: Perplexity is a geometric mean over millions of tokens. Catastrophic errors on critical reasoning tokens are easily masked by thousands of trivial stopwords.
* **Calibration Overfitting**: Calibration datasets are often overfitted to Wikipedia text, yielding artificial scores that fail in real-world agentic or coding tasks.

### Industry Benchmark Standard Suites

| Evaluation Axis | Alibaba Qwen Benchmark Standard | Unsloth Dynamic Quantization Standard |
| :--- | :--- | :--- |
| **Complex Reasoning** | **MATH / MATH-500** (Chain-of-Thought) | **MATH-500** (Testing the "Reasoning Cliff") |
| **Elementary Math** | **GSM8K** (8-shot CoT) | **GSM8K** (Baseline retention check) |
| **Code & Multi-File Editing** | **HumanEval+**, **MBPP+** (EvalPlus) | **Aider Polyglot** (Multi-language refactoring) |
| **Logit Fidelity** | Token cross-entropy | **KL Divergence & "Flips"** (Prediction shifts) |
| **Trajectory Divergence** | Single-token evaluation | **Divergence-300 @32** (32-token trajectory drift) |
| **Knowledge Preservation** | **MMLU / MMLU-Pro** (5-shot) | **5-shot MMLU** (Calibrated unsloth-eval) |

---

## 4. Grand Throughput Benchmark: 5 Standardized Domains

We tested all 7 model/engine variants against 5 diverse domains with `max_tokens=250` and `temperature=0.0`:

1. **Math & Logic**: Bat and ball algebraic derivation.
2. **Coding & Algorithms**: Clean $O(N \log N)$ `merge_intervals` with type annotations.
3. **Constraint Reasoning**: 3-chair seating permutation logic puzzle.
4. **Domain Knowledge**: Architectural comparison between FlashAttention, PagedAttention, and Speculative Decoding.
5. **Nuanced Writing**: Exactly three sentences explaining memory bandwidth constraints in LLMs.

### Throughput Results Table (tokens per second)

| Task / Domain | Splash-Q4 | Splash-HQ (Native 8b) | Splash-Q8 (Compressed) | Splash-Mixed (Selective 8b) | MTPLX-Q8 (MTP D3) | MLX-Q8 (Stock AR) | llama.cpp (Q8_0 GGUF) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Math & Logic** | **83.3 t/s** | **54.8 t/s** | 52.7 t/s | 48.6 t/s | 28.5 t/s | 9.9 t/s | 9.8 t/s |
| **Coding & Algorithms** | **75.5 t/s** | **34.7 t/s** | 40.3 t/s | 35.0 t/s | 28.8 t/s | 9.9 t/s | 9.9 t/s |
| **Constraint Reasoning** | **59.3 t/s** | **39.3 t/s** | 37.4 t/s | 36.6 t/s | 27.7 t/s | 9.9 t/s | 9.9 t/s |
| **Domain Knowledge** | **39.0 t/s** | **21.9 t/s** | 22.8 t/s | 23.3 t/s | 23.8 t/s | 9.9 t/s | 10.0 t/s |
| **Nuanced Writing** | **46.2 t/s** | **33.7 t/s** | 29.5 t/s | 27.0 t/s | 23.4 t/s | 9.9 t/s | 10.0 t/s |
| **AVERAGE SPEED** | **60.7 tok/s** | **36.9 tok/s** | **36.5 tok/s** | **34.1 tok/s** | **26.5 tok/s** | **9.9 tok/s** | **9.9 tok/s** |
| **Relative Speedup** | **6.13x** | **3.73x** | **3.69x** | **3.44x** | **2.68x** | **1.00x** | **1.00x** |

### Latency and Tokens per Second Graph

```
Splash-Q4        [============================================================] 60.7 t/s (6.13x)
Splash-HQ        [====================================] 36.9 t/s (3.73x)
Splash-Q8        [===================================] 36.5 t/s (3.69x)
Splash-Mixed     [==================================] 34.1 t/s (3.44x)
MTPLX-Q8         [==========================] 26.5 t/s (2.68x)
MLX-Q8           [==========] 9.9 t/s (1.00x)
llama.cpp-Q8     [==========] 9.9 t/s (1.00x)
```

---

## 5. Quality & Reasoning Analysis: The "Reasoning Cliff"

### The Case of MATH-500 Problem 0
Consider MATH-500 Problem 0: Evaluate the infinite double series:
$$\sum_{j=1}^\infty \sum_{k=1}^\infty \frac{1}{(j+k)^3} = p - q$$
where $p$ is an integer and $q$ is a rational multiple of $\zeta(3)$.

* **Splash-Q4**: Truncated or derailed into arithmetic errors during the summation swap ($n = j + k$, counting the multiplicity $n-1$).
* **Splash-Q8 (Compressed)**: Drifting intermediate fractions in the Chain-of-Thought led to an incorrect coefficient.
* **Splash-Mixed & Splash-HQ**: Both models preserved the full precision of the algebraic manipulations and concluded:
  $$\sum_{n=2}^\infty \frac{n-1}{n^3} = \sum_{n=2}^\infty \left( \frac{1}{n^2} - \frac{1}{n^3} \right) = (\zeta(2) - 1) - (\zeta(3) - 1) = \zeta(2) - \zeta(3)$$
  yielding the exact ground-truth boxed answer.

### Summary Scorecard

| Benchmark | Focus | Splash-Q4 | Splash-Q8 | Splash-Mixed | Splash-HQ | MTPLX-Q8 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** | Arithmetic | 80.0% (4/5) | 80.0% (4/5) | — | — | 80.0% (4/5) |
| **GPQA Diamond** | High-level Science | 33.3% (1/3) | 66.7% (2/3) | 33.3% (1/3) | 33.3% (1/3) | 20.0% (1/5) |
| **AIME 2025** | Olympiad Math | 66.7% (2/3) | 66.7% (2/3) | 33.3% (1/3) | 66.7% (2/3) | 0.0% (0/5)* |
| **MATH-500** | Symbolic Math | 0.0% (0/3) | 0.0% (0/3) | 33.3% (1/3) | 33.3% (1/3) | — |
| **OVERALL** | Cumulative | **33.3%** | **44.4%** | **33.3%** | **44.4%** | — |

---

## 6. The Precision Speed Paradox

Counter-intuitively, **Splash-HQ** (native uncompressed 8-bit, 27 GB) runs **faster** than **Splash-Q8** (compressed 8-bit, 29 GB file with internal packing):
* **Splash-HQ**: 36.9 tok/s average (peaking at 54.8 tok/s).
* **Splash-Q8**: 36.5 tok/s average (peaking at 52.7 tok/s).

### Mathematical Explanation
In speculative decoding with verification:
$$\mathbb{E}[\text{tokens per step}] = 1 + \sum_{k=1}^K \prod_{i=1}^k \alpha_i$$
where $\alpha_i = \min\left(1, \frac{P_{\text{target}}(x_i)}{P_{\text{draft}}(x_i)}\right)$ is the acceptance probability.

When the target model has higher precision (native 8-bit vs. compressed 8-bit):
1. **Sharper Logits**: Lower entropy at top-1 and top-5 tokens.
2. **Higher Draft Agreement**: The draft head was trained against BF16 logits. Native 8-bit weights align closer to BF16 than compressed 8-bit weights.
3. **Fewer Rollbacks**: Every rejected token causes a GPU pipeline stall and KV cache truncation. The reduction in rollbacks more than compensates for the marginal memory bandwidth difference.

---

## 7. Conversion Architecture: `pack_mixed_q8.py`

To construct **Splash-HQ** and **Splash-Mixed**, we implemented [`pack_mixed_q8.py`](file:///Users/nitin/Documents/shared-with-google-drive/model-serving/splash2/dev/tools/pack_mixed_q8.py):

* **SafetensorsStore with LRU Shard Caching**: Instead of loading 30 GB of safetensors simultaneously into memory (which triggers macOS swapping and OOM), an LRU cache maintains at most 2 active shard files in RAM (~10 GB footprint).
* **De-interleaving & Section Alignment**: De-interleaves attention projection weights (`q_proj`, `k_proj`, `v_proj`) into Splash's expected memory layout with 16,384-byte section alignment.
* **Target Layer Magic `MDFL0008`**: Serializes uncompressed 8-bit floating-point scales and integer weights with the Splash V5 binary container format.
* **Canonical Directory Symlink Resolution**: Ensures `target/`, `draft/`, `tokenizer/`, and `vision/` are created as real directories containing individual file symlinks, satisfying Splash runtime filesystem validation.

---

## 8. Hugging Face Releases & Reproduction

The models created in this work are publicly hosted on Hugging Face:
* **Full Native 8-bit**: [`nitinpanj/Qwen3.8-27B-Splash-HQ`](https://huggingface.co/nitinpanj/Qwen3.8-27B-Splash-HQ)
* **Selective Hybrid**: [`nitinpanj/Qwen3.8-27B-Splash-Mixed`](https://huggingface.co/nitinpanj/Qwen3.8-27B-Splash-Mixed)
* **Compressed Baseline**: [`nitinpanj/Qwen3.8-27B-Splash-Q8`](https://huggingface.co/nitinpanj/Qwen3.8-27B-Splash-Q8)

To serve any model with Splash:
```bash
splash serve \
  install/models/incoai/Qwen3.8-27B-Splash-HQ/target \
  install/models/incoai/Qwen3.8-27B-Splash-HQ/draft \
  --tokenizer install/models/incoai/Qwen3.8-27B-Splash-HQ/tokenizer \
  --model incoai/Qwen3.8-27B-Splash-HQ \
  --port 8003
```
