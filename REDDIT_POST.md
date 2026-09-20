# [Benchmarking & Weights] Qwen3.8-27B at 55 tok/s on Apple Silicon: Custom Metal kernels, native 8-bit weights, and the "Reasoning Cliff"

Over the past few days I've been profiling Qwen3.8-27B on Apple Silicon (M-series, unified memory) to see how far we can push speculative decoding and quantization precision without breaking reasoning quality.

We evaluated three runtimes across the exact same model weights: **Splash** (custom tiled Metal kernels + speculative decoding), **MTPLX** (MLX multi-token prediction runtime), and stock autoregressive baselines (**MLX** and **llama.cpp**). 

Here are the raw numbers, the weird architectural anomalies we hit (including a precision-speed paradox and a severe "reasoning cliff" on competition math), and the steps to reproduce it yourself.

---

### 1. Benchmark Results (Throughput on Unified Memory)

All benchmarks were run at `temperature=0.0`, identical system load, comparing autoregressive vs. speculative runtimes:

| Task / Domain | Prompt Description | Splash-Q4 | Splash-HQ (Native 8b) | Splash-Q8 (Compressed) | Splash-Mixed (Selective 8b) | MTPLX-Q8 (MTP D3) | Stock MLX (Q8) | llama.cpp (Q8_0) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Math & Logic** | Bat & Ball algebraic derivation | **83.3 t/s** | **54.8 t/s** | 52.7 t/s | 48.6 t/s | 28.5 t/s | 9.9 t/s | 9.8 t/s |
| **Coding & Algos** | `merge_intervals` $O(N \log N)$ | **75.5 t/s** | **34.7 t/s** | 40.3 t/s | 35.0 t/s | 28.8 t/s | 9.9 t/s | 9.9 t/s |
| **Constraint Reasoning** | 3-chair spatial permutation | **59.3 t/s** | **39.3 t/s** | 37.4 t/s | 36.6 t/s | 27.7 t/s | 9.9 t/s | 9.9 t/s |
| **Domain Knowledge** | FlashAttn vs PagedAttn vs SpecDec | **39.0 t/s** | **21.9 t/s** | 22.8 t/s | 23.3 t/s | 23.8 t/s | 9.9 t/s | 10.0 t/s |
| **Nuanced Writing** | Bandwidth constraint explanation | **46.2 t/s** | **33.7 t/s** | 29.5 t/s | 27.0 t/s | 23.4 t/s | 9.9 t/s | 10.0 t/s |
| **AVERAGE** | Across 5 standardized tasks | **60.7 t/s** | **36.9 t/s** | **36.5 t/s** | **34.1 t/s** | **26.5 t/s** | **9.9 t/s** | **9.9 t/s** |
| **Speedup vs AR** | Relative to 9.9 t/s baseline | **6.13x** | **3.73x** | **3.69x** | **3.44x** | **2.68x** | **1.00x** | **1.00x** |

**Takeaways:**
1. **Splash vs MTPLX (+39% overall, +92% math):** Both Splash-HQ and MTPLX-Q8 run on the exact same 8-bit quantized weights. But Splash’s compiled C++ Metal backend avoids Python/GIL dispatch overhead and uses custom tiled decode kernels, getting 36.9 t/s vs MTPLX's 26.5 t/s. On structured math derivation, Splash-HQ hit **54.8 tok/s**.
2. **Splash-Q4 is fast:** If 4-bit quantization error is acceptable for your use case, Splash-Q4 sustains ~60 tok/s on a 27B model, peaking over 83 tok/s.

---

### 2. The "Reasoning Cliff" on Competition Math

Throughput is useless if the model hallucinates math derivations. We evaluated long chain-of-thought accuracy on AIME 2025, GPQA Diamond, and MATH-500:

* On **MATH-500 Problem 0** (evaluating $\sum_{j=1}^\infty \sum_{k=1}^\infty \frac{1}{(j+k)^3}$), both stock **Splash-Q4** and compressed **Splash-Q8** fell off a cliff: they suffered numerical drift halfway through the derivation and produced incorrect closed-form solutions.
* When we upgraded the model to **Splash-HQ** (uncompressed native 8-bit across all 64 layers) or **Splash-Mixed** (where only the top 8 sensitive layers, 56–63, are 8-bit), the model solved the problem completely and cleanly derived $p - q$.
* Upgrading just the deepest 8 layers resolved the reasoning breakdown while keeping the model memory footprint lean.

---

### 3. The Precision-Speed Paradox in Speculative Decoding

Here's something counterintuitive: **Splash-HQ (native uncompressed 8-bit, 27 GB) ran faster on average than Splash-Q8 (compressed 8-bit, 17 GB)**—36.9 tok/s vs 36.5 tok/s.

Why? In speculative decoding, decode speed is a product of:
$$\text{Speed} = \text{Draft Speed} \times \text{Acceptance Rate}$$

When target weights are quantized too aggressively, logit distributions flatten and drift. Even a 5% drop in draft acceptance rate forces more speculative rollbacks and target re-verifications. With native 8-bit weights, sharper logits boosted the draft acceptance rate enough to more than offset the extra bytes fetched from RAM.

---

### 4. How to Run It

Upstream Splash (`incoai/splash`) only has 4-bit package formats hardcoded (`splash-packed-q4`, schema 3/4). 

To run the native 8-bit model (`splash-packed-q8`, schema 5), you need the Q8-enabled runtime fork with compiled Metal Q8 tiled kernels. It is completely backwards-compatible and also runs standard Q4 models.

#### Step 1: Clone and build the runtime
```bash
git clone https://github.com/npanj/splash.git -b q8
cd splash
make -j4
```

#### Step 2: Launch the server
This will download the 27 GB model weights from Hugging Face on first run and start an OpenAI-compatible endpoint on port 8000:
```bash
./splash serve --model nitinpanj/Qwen3.8-27B-Splash-HQ --port 8000
```

*(Note: If you have already downloaded the weights, you can also point directly to the local directory).*

#### Step 3: Query via curl or Python
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nitinpanj/Qwen3.8-27B-Splash-HQ",
    "messages": [{"role": "user", "content": "Explain why uncompressed 8-bit weights can yield higher speculative decoding throughput than compressed weights."}],
    "temperature": 0.0
  }'
```

---

### Links & Artifacts

* **Hugging Face Model (Full 27 GB Weights)**: [`nitinpanj/Qwen3.8-27B-Splash-HQ`](https://huggingface.co/nitinpanj/Qwen3.8-27B-Splash-HQ)
* **Q8 Metal Runtime Fork**: [`https://github.com/npanj/splash/tree/q8`](https://github.com/npanj/splash/tree/q8)
* **Full Benchmark Scripts & Raw JSON Logs**: [`https://github.com/npanj/qwen3.8-27b-apple-silicon-eval`](https://github.com/npanj/qwen3.8-27b-apple-silicon-eval)

Let me know if you run into any issues testing it on your M-series setup.
