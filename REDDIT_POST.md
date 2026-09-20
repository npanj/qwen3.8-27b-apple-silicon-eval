Spent the last 24 hours building, packaging, and benchmarking Qwen3.8-27B in native 8-bit on Apple Silicon (M5 Pro, 64 GB unified memory).

Standard autoregressive serving (stock MLX, llama.cpp Q8_0) caps out at around 9.9 tok/s on a 27B model because you're reading ~27 GB of weights on every single token. While 4-bit speculative decoding (Splash-Q4) is fast (~60 tok/s), aggressive 4-bit quants hit a nasty "reasoning cliff" on competition-grade math and multi-step derivations.

We wanted true uncompressed 8-bit quality without giving up speculative decoding speed. Finally got the pipeline, custom Metal kernels, and evaluation suite working cleanly.

* **Model weights (27 GB native 8-bit):** https://huggingface.co/nitinpanj/Qwen3.8-27B-Splash-HQ
* **Q8 Metal runtime fork:** https://github.com/npanj/splash/tree/q8
* **Benchmark suite & raw JSON logs:** https://github.com/npanj/qwen3.8-27b-apple-silicon-eval
* **Benchmark & Context Scaling Plot:** https://raw.githubusercontent.com/npanj/qwen3.8-27b-apple-silicon-eval/main/benchmark_and_context_scaling.png

This won't run on upstream Splash 1.0 (`incoai/splash`). Upstream hardcodes package validation to only accept 4-bit schemas (`splash-packed-q4`, schema 3/4) and rejects 8-bit packages. The fork adds schema 5 (`splash-packed-q8`, `MDFL0008`) loading, custom Metal tiled Q8 decode kernels, and maintains 100% backwards compatibility with stock Q4 models.

---

### Speeds on Apple Silicon (M5 Pro, 64 GB Unified Memory)

Evaluated at `temperature=0.0` across 5 standardized task domains:

| Task / Domain | Prompt Description | Splash-Q4 (4b MTP) | Splash-HQ (Native 8b) | Splash-Q8 (Compressed) | MTPLX-Q8 (MTP D3) | Stock MLX / llama.cpp (AR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Math & Logic** | Algebraic derivation | **83.3 t/s** | **54.8 t/s** | 52.7 t/s | 28.5 t/s | 9.9 t/s |
| **Coding & Algos** | `merge_intervals` $O(N \log N)$ | **75.5 t/s** | **34.7 t/s** | 40.3 t/s | 28.8 t/s | 9.9 t/s |
| **Constraint Reasoning**| 3-chair spatial permutation | **59.3 t/s** | **39.3 t/s** | 37.4 t/s | 27.7 t/s | 9.9 t/s |
| **Domain Knowledge** | FlashAttn vs PagedAttn | **39.0 t/s** | **21.9 t/s** | 22.8 t/s | 23.8 t/s | 9.9 t/s |
| **Nuanced Writing** | Memory bandwidth constraint | **46.2 t/s** | **33.7 t/s** | 29.5 t/s | 23.4 t/s | 9.9 t/s |
| **AVERAGE** | **Across all 5 domains** | **60.7 t/s** | **36.9 t/s** | **36.5 t/s** | **26.5 t/s** | **9.9 t/s** |
| **Speedup vs AR** | Relative to 9.9 t/s baseline | **6.13x** | **3.73x** | **3.69x** | **2.68x** | **1.00x** |

**A few notes on the comparisons:**
* **Splash-HQ vs MTPLX (+39% overall, +92% math):** Both run on the exact same 8-bit base weights. But Splash’s compiled C++ Metal backend executes with lower dispatch overhead than Python/MLX DraftCore, getting 36.9 vs 26.5 tok/s overall, and hitting **54.8 tok/s** on structured math reasoning.
* **The Precision-Speed Paradox:** Uncompressed native 8-bit (Splash-HQ, 27 GB) actually ran slightly *faster* on average than compressed 8-bit (Splash-Q8, 17 GB)—36.9 vs 36.5 tok/s. In speculative decoding, decode speed is $\text{Draft Speed} \times \text{Acceptance Rate}$. Aggressive compression flattened logits and lowered draft acceptance; native 8-bit produced sharper logits, fewer verification rollbacks, and higher net throughput despite reading more bytes from memory.

---

### Context Scaling: What Happens from 0 to 190,000 Tokens

Most Transformers fall off a cliff in decode speed as context grows because the KV cache balloons. 

Qwen3.8 uses a hybrid architecture: **48 recurrent linear DeltaNet layers** (fixed $128 \times 128$ hidden state, $O(1)$ memory growth with context) and only **16 full-attention layers**. 

Here is raw telemetry sampled from my live server session as the context grew from scratch all the way to 190k tokens:

| Context Length (Tokens) | Cached Tokens | Generated Output | TTFT (Prompt Prefill) | **Decode Speed** | Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **65** | 0 | 50 | 0.8s | **35.7 tok/s** | Short prompt baseline |
| **16,433** | 15,040 | 232 | 4.0s | **49.0 tok/s** | Prefix cache hit |
| **34,605** | 29,376 | 2,771 | 15.6s | **27.1 tok/s** | Long response generation |
| **83,379** | 76,320 | 435 | 28.9s | **30.1 tok/s** | Deep context code review |
| **106,212** | 98,752 | 29,487 | 34.0s | **24.8 tok/s** | Massive batch generation |
| **157,961** | 157,056 | 400 | 6.1s | **43.5 tok/s** | Cache hit at 158k tokens |
| **180,082** | 143,360 | 3,446 | 228.5s | **33.3 tok/s** | Extended reasoning session |
| **187,613** | 186,720 | 425 | 6.5s | **31.9 tok/s** | Cache hit at 187k tokens |
| **188,546** | 147,456 | 1,083 | 268.2s | **21.1 tok/s** | Partial prefill recompute |
| **190,016** | 151,552 | 1,115 | 227.4s | **32.0 tok/s** | Max context reached |

*(See the visual plot in the repo: [benchmark_and_context_scaling.png](https://raw.githubusercontent.com/npanj/qwen3.8-27b-apple-silicon-eval/main/benchmark_and_context_scaling.png) showing the full 51-point scatter and rolling trend line).*

**The big takeaway on context:**
Decode speed **does not collapse**. It stays between **21 – 33 tok/s** all the way out to 190k tokens. 

The actual bottleneck at 150k+ context is **cold prefill (TTFT)**. When the prefix cache hits, TTFT at 187k context is just **6.5 seconds**. But on a cold cache miss, prefilling 180k+ tokens on a 27B model on Apple Silicon takes ~4–5 minutes. If you are using agent harnesses (like Oh My Pi, Claude Code, or curl), make sure client SSE idle timeouts are set high enough so the client doesn't drop the connection during cold prefills.

---

### The "Reasoning Cliff" on Competition Math

Throughput numbers don't matter if math derivations hallucinate. We tested extended CoT reasoning on MATH-500, AIME 2025, and GPQA Diamond:

* On **MATH-500 Problem 0** (evaluating $\sum_{j=1}^\infty \sum_{k=1}^\infty \frac{1}{(j+k)^3} = p - q$), both stock **Splash-Q4** and compressed **Splash-Q8** fell off a cliff: they suffered numerical drift halfway through the algebraic series manipulation and output wrong values.
* Upgrading to **Splash-HQ** (full uncompressed 8-bit across all 64 layers) or **Splash-Mixed** (where only the top 8 sensitive layers, 56–63, are 8-bit) completely eliminated the cliff and cleanly derived $p - q$.
* Upgrading just the deepest 8 layers restored the full symbolic precision while keeping RAM manageable.

---

### Setup Recipe (3 Steps)

#### 1. Clone the Q8 runtime fork
```bash
git clone https://github.com/npanj/splash.git -b q8
cd splash
make -j4
```

#### 2. Start the server
```bash
./splash serve --model nitinpanj/Qwen3.8-27B-Splash-HQ --port 8000
```
*(On first run, it automatically downloads and verifies the 27 GB model artifacts into `install/models/nitinpanj/Qwen3.8-27B-Splash-HQ`).*

#### 3. Connect your client
The server exposes a standard OpenAI-compatible `/v1/chat/completions` endpoint:
```bash
# Test with curl
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "incoai/Qwen3.8-27B-Splash-HQ",
    "messages": [{"role": "user", "content": "Explain why uncompressed 8-bit weights improve speculative decoding acceptance."}],
    "temperature": 0.0
  }'

# Or connect Oh My Pi (OMP)
omp --model splash/incoai/Qwen3.8-27B-Splash-HQ
```

---

### Practical Gotchas & Details
1. **Memory headroom at 150k+ context:** On a 64 GB Mac, model weights take ~27 GB. As context pushes towards 180k–190k, working memory climbs to ~42 GiB. Metal's memory governor will pause allocation growth when system free RAM dips below ~50 MB (`Memory: growth paused`). If you don't need 190k context, you can pass `--max-context 131072` to cap it cleanly.
2. **Backwards compatibility:** You don't need two binaries. This fork preserves all upstream 4-bit dense and MoE schemas (`splash-packed-q4`, `splash-packed-q4-moe`), so you can serve official models like `incoai/Qwen3.8-27B-Splash` or `incoai/Qwen3.6-35B-A3B-Splash` directly.
3. **Only tested on Apple Silicon (Unified Memory):** Everything here relies on unified memory bandwidth and Metal tiled shaders; not tested on CUDA or CPU.

Credit where it's due: Incoai for the Splash engine and speculative decoding architecture, the Qwen team for base weights and MTP, and Youssofal for MTPLX references.
