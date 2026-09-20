#!/usr/bin/env python3
"""Benchmark all 4 Splash variants (Q4, Q8, Mixed, HQ) on the exact 5 prompts used by MTPLX, MLX, and llama.cpp."""

import json
import time
import urllib.request
import urllib.error
import os

PROMPTS = [
    {
        "id": "math_logic",
        "category": "Math & Logic",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your step-by-step algebraic derivation, followed by the final answer."
    },
    {
        "id": "coding",
        "category": "Coding & Algorithms",
        "prompt": "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges overlapping intervals. It must be clean, optimal in O(N log N) time, handle empty lists and negative coordinates, and include type annotations and docstring."
    },
    {
        "id": "spatial_constraint",
        "category": "Constraint Reasoning",
        "prompt": "Three friends (Alice, Bob, Charlie) are sitting in a row of 3 chairs numbered 1 to 3 from left to right. Constraints:\n1. Alice never sits next to Bob.\n2. Charlie is to the right of Alice (higher chair number).\nWho sits in chair 1, chair 2, and chair 3? Give a logical deduction step by step."
    },
    {
        "id": "technical_explanation",
        "category": "Domain Knowledge",
        "prompt": "Explain clearly the respective roles and architectural differences between FlashAttention, PagedAttention, and Speculative Decoding in high-performance LLM inference engines."
    },
    {
        "id": "executive_summary",
        "category": "Nuanced Writing",
        "prompt": "In exactly three clear, professional sentences, explain why memory bandwidth (rather than raw compute TFLOPS) is the dominant latency bottleneck during autoregressive token generation in transformer LLMs."
    }
]

SPLASH_SERVERS = [
    {
        "name": "Splash-Q4",
        "url": "http://127.0.0.1:8000/v1/chat/completions",
        "model": "incoai/Qwen3.8-27B-Splash",
    },
    {
        "name": "Splash-Q8",
        "url": "http://127.0.0.1:8001/v1/chat/completions",
        "model": "incoai/Qwen3.8-27B-Splash-Q8",
    },
    {
        "name": "Splash-Mixed",
        "url": "http://127.0.0.1:8004/v1/chat/completions",
        "model": "incoai/Qwen3.8-27B-Splash-Mixed",
    },
    {
        "name": "Splash-HQ",
        "url": "http://127.0.0.1:8003/v1/chat/completions",
        "model": "incoai/Qwen3.8-27B-Splash-HQ",
    },
]

def query_splash(server, prompt, max_tokens=250):
    payload = {
        "model": server["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        server["url"],
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            t1 = time.perf_counter()
            body = json.loads(raw)
            choice = body["choices"][0]["message"]
            content = choice.get("content") or ""
            reasoning = choice.get("reasoning_content") or ""
            metrics = body.get("metrics", {})
            lat = metrics.get("request_latency", {})
            tok_per_sec = lat.get("stream_tokens_per_second")
            usage = body.get("usage", {})
            comp_tokens = usage.get("completion_tokens", 0)
            if not tok_per_sec:
                tok_per_sec = comp_tokens / (t1 - t0) if (t1 - t0) > 0 else 0.0
            return {
                "content": content,
                "reasoning": reasoning,
                "text": content if content else reasoning,
                "tok_per_sec": tok_per_sec,
                "elapsed_s": t1 - t0,
                "completion_tokens": comp_tokens,
                "prompt_tokens": usage.get("prompt_tokens", 0),
            }
    except Exception as e:
        return {"error": str(e), "tok_per_sec": 0.0, "elapsed_s": 0.0, "completion_tokens": 0}

def main():
    print("=" * 80)
    print("RUNNING 5 STANDARD PROMPTS ON ALL SPLASH VARIANTS (max_tokens=250)")
    print("=" * 80)

    results = {s["name"]: [] for s in SPLASH_SERVERS}

    for item in PROMPTS:
        pid = item["id"]
        cat = item["category"]
        print(f"\nPrompt: [{cat}] {pid}")
        for server in SPLASH_SERVERS:
            sname = server["name"]
            res = query_splash(server, item["prompt"], max_tokens=250)
            results[sname].append({
                "id": pid,
                "category": cat,
                **res
            })
            if "error" in res:
                print(f"  {sname:15s}: ERROR ({res['error']})")
            else:
                print(f"  {sname:15s}: {res['tok_per_sec']:5.1f} tok/s | {res['completion_tokens']} tok in {res['elapsed_s']:.2f}s")

    out_path = "/tmp/splash_all_variants_5prompts.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Splash results to {out_path}")

    # Now load MTPLX, MLX, llama.cpp and produce grand comparison table
    print("\n" + "=" * 90)
    print("GRAND COMPARISON ACROSS ALL 7 ENGINES ON 5 IDENTICAL PROMPTS (max_tokens=250)")
    print("=" * 90)

    # Load external results
    with open("/tmp/mtplx_q8_results.json") as f:
        mtplx_raw = json.load(f)["results"]
        mtplx_by_id = {r["id"]: r["decode_tok_s"] for r in mtplx_raw}

    with open("/tmp/mlx_q8_results.json") as f:
        mlx_raw = json.load(f)["results"]
        mlx_by_id = {r["id"]: r["generation_speed_tps"] for r in mlx_raw}

    with open("/tmp/llamacpp_q8_results.json") as f:
        llama_raw = json.load(f)
        llama_by_id = {r["id"]: r["gen_tok_per_sec"] for r in llama_raw}

    print(f"{'Task / Prompt':<22} | {'Splash-Q4':<10} | {'Splash-Q8':<10} | {'Splash-Mix':<10} | {'Splash-HQ':<10} | {'MTPLX-Q8':<10} | {'MLX-Q8':<8} | {'llama.cpp':<8}")
    print("-" * 105)

    all_speeds = {
        "Splash-Q4": [],
        "Splash-Q8": [],
        "Splash-Mixed": [],
        "Splash-HQ": [],
        "MTPLX-Q8": [],
        "MLX-Q8": [],
        "llama.cpp": []
    }

    for item in PROMPTS:
        pid = item["id"]
        row = [f"{pid:<22}"]
        for s in SPLASH_SERVERS:
            sname = s["name"]
            spd = next(r["tok_per_sec"] for r in results[sname] if r["id"] == pid)
            all_speeds[sname].append(spd)
            row.append(f"{spd:6.1f} t/s")
        
        m_spd = mtplx_by_id.get(pid, 0.0)
        all_speeds["MTPLX-Q8"].append(m_spd)
        row.append(f"{m_spd:6.1f} t/s")

        mlx_spd = mlx_by_id.get(pid, 0.0)
        all_speeds["MLX-Q8"].append(mlx_spd)
        row.append(f"{mlx_spd:5.1f} t/s")

        l_spd = llama_by_id.get(pid, 0.0)
        all_speeds["llama.cpp"].append(l_spd)
        row.append(f"{l_spd:5.1f} t/s")

        print(" | ".join(row))

    print("-" * 105)
    avg_row = [f"{'AVERAGE TOK/S':<22}"]
    for model_name in ["Splash-Q4", "Splash-Q8", "Splash-Mixed", "Splash-HQ", "MTPLX-Q8", "MLX-Q8", "llama.cpp"]:
        spds = all_speeds[model_name]
        avg = sum(spds) / len(spds) if spds else 0.0
        avg_row.append(f"{avg:6.1f} t/s")
    print(" | ".join(avg_row))
    print("=" * 105)

if __name__ == "__main__":
    main()
