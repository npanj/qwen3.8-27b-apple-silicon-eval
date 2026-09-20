#!/usr/bin/env python3
"""Benchmark MTPLX Q8 (Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality) with native MTP."""

import json
import os
import subprocess
import sys
import time

MTPLX_BIN = "/Users/nitin/Documents/shared-with-google-drive/model-serving/local-mlx/.mlx-env/bin/mtplx"
MODEL_PATH = os.path.expanduser("~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Quality")
OUTPUT_JSON = "/tmp/mtplx_q8_results.json"

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

def run_prompt(item, max_tokens=250):
    cmd = [
        MTPLX_BIN,
        "run",
        "--model", MODEL_PATH,
        "--prompt", item["prompt"],
        "--max-tokens", str(max_tokens),
        "--temperature", "0.0",
        "--mtp",
        "--json"
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    
    stdout = proc.stdout
    stderr = proc.stderr
    
    # Returncode 4 is an MTPLX validation warning (e.g. truncated delimiters at max_tokens limit),
    # which still contains the complete generation and stats JSON.
    if proc.returncode not in (0, 4) and not stdout.strip().endswith("}"):
        print(f"Error running MTPLX: returncode {proc.returncode}\n{stderr}", file=sys.stderr)
        return None
        
    # Extract json blob from stdout (skip prefix logs and prewarm json)
    json_start = stdout.find("\n{\n  \"draft_lm_head\"")
    if json_start == -1:
        # Fallback: look for the last line starting with {
        idx = stdout.rfind("\n{\n")
        if idx != -1:
            json_start = idx + 1
        else:
            json_start = stdout.find("{")
    else:
        json_start += 1
        
    if json_start == -1:
        print(f"Could not find JSON in MTPLX output:\n{stdout}", file=sys.stderr)
        return None
        
    json_str = stdout[json_start:]
    try:
        data = json.loads(json_str)
        stats = data.get("stats", {})
        return {
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "decode_tok_s": stats.get("decode_tok_s", 0.0),
            "prompt_eval_time_s": stats.get("prompt_eval_time_s", 0.0),
            "end_to_end_tok_s": stats.get("end_to_end_tok_s", 0.0),
            "generated_tokens": stats.get("generated_tokens", 0),
            "generation_mode": stats.get("generation_mode", "unknown"),
            "accepted_by_depth": stats.get("accepted_by_depth", []),
            "total_elapsed_s": t1 - t0,
            "text": data.get("text", "")
        }
    except Exception as e:
        print(f"Error parsing JSON: {e}\nRaw string:\n{json_str}", file=sys.stderr)
        return None

def main():
    print("="*80)
    print("STARTING BENCHMARK: MTPLX Q8 (Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality)")
    print("="*80 + "\n")
    
    results = []
    for i, item in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] Running: {item['category']} ({item['id']})...", flush=True)
        res = run_prompt(item)
        if res:
            results.append(res)
            print(f"  Decode Speed: {res['decode_tok_s']:.2f} t/s | Mode: {res['generation_mode']} | Gen Tokens: {res['generated_tokens']}")
            print(f"  Acceptance:   {res['accepted_by_depth']}")
            preview = res['text'][:150].replace('\n', ' ')
            print(f"  Preview:      {preview}...\n")
        else:
            print("  FAILED!\n")
            
    if results:
        avg_decode = sum(r["decode_tok_s"] for r in results) / len(results)
        print("="*80)
        print(f"MTPLX Q8 SUMMARY: Avg Decode Speed = {avg_decode:.2f} t/s")
        print("="*80)
        
        with open(OUTPUT_JSON, "w") as f:
            json.dump({
                "model": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
                "avg_decode_tok_s": avg_decode,
                "results": results
            }, f, indent=2)
        print(f"\nSaved results to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
