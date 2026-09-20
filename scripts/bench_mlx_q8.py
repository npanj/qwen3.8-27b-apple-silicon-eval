#!/usr/bin/env python3
"""Run MLX 8-bit model benchmark on the standard 5 prompts."""

import json
import os
import sys
import time
import mlx.core as mx
import mlx_lm
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Qwen3.8-27B-8bit"
OUTPUT_JSON = "/tmp/mlx_q8_results.json"

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

def main():
    print(f"[{time.strftime('%H:%M:%S')}] Loading model: {MODEL_ID}...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(MODEL_ID)
    t1 = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded in {t1 - t0:.2f}s", flush=True)

    sampler = make_sampler(temp=0.0)
    results = []

    print("\n" + "="*80)
    print(f"BENCHMARKING MLX Q8 ({MODEL_ID})")
    print("="*80 + "\n")

    for i, item in enumerate(PROMPTS, 1):
        pid = item["id"]
        category = item["category"]
        prompt = item["prompt"]

        print(f"--- [{i}/{len(PROMPTS)}] {category} ({pid}) ---")

        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        mx.reset_peak_memory()
        t_start = time.perf_counter()
        
        gen_stream = stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=250,
            sampler=sampler
        )

        response_chunks = []
        prompt_tokens = 0
        prompt_tps = 0.0
        gen_tokens = 0
        gen_tps = 0.0
        peak_mem = 0.0

        for r in gen_stream:
            response_chunks.append(r.text)
            prompt_tokens = r.prompt_tokens
            prompt_tps = r.prompt_tps
            gen_tokens = r.generation_tokens
            gen_tps = r.generation_tps
            peak_mem = r.peak_memory

        t_total = time.perf_counter() - t_start
        full_text = "".join(response_chunks).strip()

        print(f"  Prompt tokens: {prompt_tokens} | Prompt speed: {prompt_tps:.2f} t/s")
        print(f"  Gen tokens:    {gen_tokens} | Gen speed:    {gen_tps:.2f} t/s")
        print(f"  Total time:    {t_total:.2f}s | Peak VRAM:     {peak_mem:.2f} GB")
        print("  Preview:")
        preview = full_text[:200].replace('\n', ' ')
        print(f"    {preview}...\n")

        results.append({
            "id": pid,
            "category": category,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "prompt_speed_tps": prompt_tps,
            "generation_tokens": gen_tokens,
            "generation_speed_tps": gen_tps,
            "total_time_s": t_total,
            "peak_memory_gb": peak_mem,
            "output": full_text
        })

    avg_gen_tps = sum(r["generation_speed_tps"] for r in results) / len(results) if results else 0
    avg_prompt_tps = sum(r["prompt_speed_tps"] for r in results) / len(results) if results else 0
    max_peak_mem = max(r["peak_memory_gb"] for r in results) if results else 0

    print("="*80)
    print(f"MLX Q8 SUMMARY: Avg Gen Speed = {avg_gen_tps:.2f} t/s | Avg Prompt Speed = {avg_prompt_tps:.2f} t/s | Peak Memory = {max_peak_mem:.2f} GB")
    print("="*80)

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "avg_generation_speed_tps": avg_gen_tps,
            "avg_prompt_speed_tps": avg_prompt_tps,
            "peak_memory_gb": max_peak_mem,
            "results": results
        }, f, indent=2)
    print(f"\nSaved results to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
