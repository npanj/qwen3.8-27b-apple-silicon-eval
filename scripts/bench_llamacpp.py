#!/usr/bin/env python3
"""Run llama-cli on the AtomicChat Qwen3.8-27B-Q8_0 model with the same benchmark prompts."""

import json
import re
import subprocess
import time

LLAMA_CLI = "/Users/nitin/Documents/shared-with-google-drive/model-serving/llama.cpp-latest/build/bin/llama-cli"
MODEL_PATH = "/Users/nitin/Library/Application Support/Atomic Chat/data/llamacpp/models/AtomicChat/Qwen3_8-27B-Q8_0/model.gguf"

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

def run_prompt(prompt, max_tokens=250):
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", str(max_tokens),
        "-ngl", "99",
        "--temp", "0.0",
        "-st",
        "--simple-io"
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    
    output = proc.stdout + "\n" + proc.stderr
    
    # Extract Prompt t/s and Generation t/s
    prompt_speed = 0.0
    gen_speed = 0.0
    m = re.search(r"Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s", output)
    if m:
        prompt_speed = float(m.group(1))
        gen_speed = float(m.group(2))
        
    return {
        "elapsed": t1 - t0,
        "prompt_tok_per_sec": prompt_speed,
        "gen_tok_per_sec": gen_speed,
        "output": output
    }

def main():
    print("=" * 80)
    print("RUNNING LLAMA.CPP / ATOMICCHAT Q8_0 BENCHMARK")
    print("=" * 80)
    
    results = []
    for item in PROMPTS:
        print(f"\nPrompt [{item['id']}]: Running...")
        res = run_prompt(item["prompt"])
        print(f"  Prompt speed: {res['prompt_tok_per_sec']:.2f} t/s | Generation speed: {res['gen_tok_per_sec']:.2f} t/s | Wall clock: {res['elapsed']:.2f}s")
        results.append({
            "id": item["id"],
            "category": item["category"],
            "prompt_tok_per_sec": res["prompt_tok_per_sec"],
            "gen_tok_per_sec": res["gen_tok_per_sec"],
            "elapsed": res["elapsed"],
        })
        
    out_file = "/tmp/llamacpp_q8_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
