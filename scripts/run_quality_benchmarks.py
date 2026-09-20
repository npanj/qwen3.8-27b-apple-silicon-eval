#!/usr/bin/env python3
"""Run automated quality benchmarks across Splash Q8, Splash Q4, and MTPLX Q8."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error

# Optional parquet reader for standard benchmarks (GSM8K, MMLU)
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

LOCAL_BENCH_DIR = "/Users/nitin/Documents/shared-with-google-drive/model-serving/local-mlx/bench_data"
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
MTPLX_BIN = "/Users/nitin/Documents/shared-with-google-drive/model-serving/local-mlx/.mlx-env/bin/mtplx"
MTPLX_MODEL = os.path.expanduser("~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Quality")

# Dataset definitions
def load_gpqa(limit=10):
    path = os.path.join(LOCAL_BENCH_DIR, "gpqa.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for idx, row in enumerate(data[:limit]):
        prompt = (
            "Answer the following multiple-choice question. Think step by step and conclude your "
            'response with: "Therefore, the correct answer is (X)" where X is one of A, B, C, or D.\n\n'
            f"Question:\n{row['question']}"
        )
        items.append({
            "id": f"gpqa_{idx}",
            "benchmark": "GPQA Diamond",
            "type": "choice",
            "prompt": prompt,
            "ground_truth": row["answer"].strip().upper(),
        })
    return items

def load_aime25(limit=10):
    path = os.path.join(LOCAL_BENCH_DIR, "aime25.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for idx, row in enumerate(data[:limit]):
        prompt = (
            "Solve the following math competition problem. The answer is a non-negative integer from 0 to 999. "
            "Think step by step, keep your derivation concise, and conclude your response with: "
            '"Therefore, the final answer is \\boxed{N}" where N is an integer from 0 to 999.\n\n'
            f"Problem:\n{row['problem']}"
        )
        items.append({
            "id": f"aime25_{idx}",
            "benchmark": "AIME 2025",
            "type": "integer",
            "prompt": prompt,
            "ground_truth": str(int(row["answer"].strip())),
        })
    return items

def load_math500(limit=10):
    path = os.path.join(LOCAL_BENCH_DIR, "math500.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for idx, row in enumerate(data[:limit]):
        prompt = (
            f"Solve the following mathematics problem about {row.get('subject', 'math')}. "
            "Think step by step and conclude your response with: "
            '"Therefore, the final answer is \\boxed{ANSWER}"\n\n'
            f"Problem:\n{row['problem']}"
        )
        items.append({
            "id": f"math500_{idx}",
            "benchmark": "MATH-500",
            "type": "math",
            "prompt": prompt,
            "ground_truth": row["answer"].strip(),
        })
    return items

def load_gsm8k(limit=10):
    if pq is None:
        return []
    path = os.path.join(
        HF_CACHE_DIR,
        "datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet"
    )
    if not os.path.exists(path):
        return []
    tbl = pq.read_table(path)
    items = []
    count = min(limit, len(tbl))
    for idx in range(count):
        q = tbl["question"][idx].as_py()
        ans_raw = tbl["answer"][idx].as_py()
        m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ans_raw)
        gt = m.group(1).replace(",", "") if m else ""
        prompt = (
            "Solve the following math word problem. Think step by step and conclude your "
            'response with: "Therefore, the final answer is \\boxed{N}" where N is the final numerical value.\n\n'
            f"Question:\n{q}"
        )
        items.append({
            "id": f"gsm8k_{idx}",
            "benchmark": "GSM8K",
            "type": "numeric",
            "prompt": prompt,
            "ground_truth": gt,
        })
    return items

def load_mmlu(limit=10):
    if pq is None:
        return []
    path = os.path.join(
        HF_CACHE_DIR,
        "datasets--cais--mmlu/snapshots/c30699e8356da336a370243923dbaf21066bb9fe/all/test-00000-of-00001.parquet"
    )
    if not os.path.exists(path):
        return []
    tbl = pq.read_table(path)
    letters = ["A", "B", "C", "D"]
    items = []
    count = min(limit, len(tbl))
    for idx in range(count):
        q = tbl["question"][idx].as_py()
        subject = tbl["subject"][idx].as_py()
        choices = tbl["choices"][idx].as_py()
        ans_idx = int(tbl["answer"][idx].as_py())
        gt = letters[ans_idx]
        
        choices_text = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(min(4, len(choices))))
        prompt = (
            f"The following is a multiple-choice question about {subject.replace('_', ' ')}. "
            'Think step by step and conclude your response with: "Therefore, the correct answer is (X)" '
            "where X is one of A, B, C, or D.\n\n"
            f"Question:\n{q}\n\n{choices_text}"
        )
        items.append({
            "id": f"mmlu_{idx}",
            "benchmark": "MMLU",
            "type": "choice",
            "prompt": prompt,
            "ground_truth": gt,
        })
    return items

# Answer extraction
def extract_answer(text: str, ans_type: str) -> str | None:
    if not text:
        return None
    if ans_type == "choice":
        patterns = [
            r"(?:correct answer is|answer is|correct option is|choice is)\s*[:\*\s]*\(?([A-D])\)?\b",
            r"\\boxed\{([A-D])\}",
            r"(?:^|\n)\s*([A-D])\s*$",
            r"\b([A-D])\b",
        ]
        for pat in patterns:
            matches = list(re.finditer(pat, text, re.IGNORECASE))
            if matches:
                return matches[-1].group(1).upper()
        return None
    elif ans_type in ("integer", "numeric", "math"):
        boxed = list(re.finditer(r"\\boxed\{([^{}]+)\}", text))
        if boxed:
            val = boxed[-1].group(1).replace(",", "").replace("$", "").replace("\\$", "").strip()
            if ans_type == "integer":
                try:
                    return str(int(float(val)))
                except ValueError:
                    return val
            return val
        patterns = [
            r"(?:final answer is|answer is|total is|equals)\s*[:\*\s]*[$]?(-?[\d,]+(?:\.\d+)?)[$]?",
            r"####\s*(-?[\d,]+(?:\.\d+)?)",
            r"(?:^|\n)\s*(-?[\d,]+(?:\.\d+)?)\s*$",
        ]
        for pat in patterns:
            matches = list(re.finditer(pat, text, re.IGNORECASE))
            if matches:
                val = matches[-1].group(1).replace(",", "").replace("$", "").replace("\\$", "").strip()
                if ans_type == "integer":
                    try:
                        return str(int(float(val)))
                    except ValueError:
                        return val
                return val
        return None
    return None

def grade_answer(prediction: str | None, ground_truth: str) -> bool:
    if prediction is None or not ground_truth:
        return False
    pred = prediction.strip().upper().replace("$", "").replace("\\$", "").replace(" ", "")
    gt = ground_truth.strip().upper().replace("$", "").replace("\\$", "").replace(" ", "")
    if pred == gt:
        return True
    try:
        if abs(float(pred) - float(gt)) < 1e-5:
            return True
    except ValueError:
        pass
    return False

# Model runners
def run_splash_http(port: int, model_name: str, prompt: str, max_tokens: int = 1024):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
            t1 = time.perf_counter()
            body = json.loads(raw)
            choice = body["choices"][0]["message"]
            content = choice.get("content") or ""
            reasoning = choice.get("reasoning_content") or ""
            usage = body.get("usage", {})
            metrics = body.get("metrics", {})
            lat = metrics.get("request_latency", {})
            tok_per_sec = lat.get("stream_tokens_per_second")
            if not tok_per_sec:
                comp_tokens = usage.get("completion_tokens", 0)
                tok_per_sec = comp_tokens / (t1 - t0) if (t1 - t0) > 0 else 0.0
            return {
                "text": content if content else reasoning,
                "content": content,
                "reasoning": reasoning,
                "elapsed_s": t1 - t0,
                "tok_s": tok_per_sec,
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
            }
    except Exception as e:
        return {"error": str(e), "text": "", "content": "", "reasoning": "", "elapsed_s": 0.0, "tok_s": 0.0}

def run_mtplx(prompt: str, max_tokens: int = 400):
    # Try HTTP first if mtplx server is up on port 8002
    try:
        url = "http://127.0.0.1:8002/v1/chat/completions"
        payload = {
            "model": "mtplx-qwen38-27b-optimized-quality",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            t1 = time.perf_counter()
            body = json.loads(raw)
            choice = body["choices"][0]["message"]
            content = choice.get("content") or ""
            reasoning = choice.get("reasoning_content") or ""
            usage = body.get("usage", {})
            comp_tokens = usage.get("completion_tokens", 0)
            tok_per_sec = comp_tokens / (t1 - t0) if (t1 - t0) > 0 else 0.0
            return {
                "text": content if content else reasoning,
                "reasoning": reasoning,
                "elapsed_s": t1 - t0,
                "tok_s": tok_per_sec,
                "completion_tokens": comp_tokens,
                "prompt_tokens": usage.get("prompt_tokens", 0),
            }
    except Exception:
        # Fall back to CLI
        pass

    cmd = [
        MTPLX_BIN,
        "run",
        "--model", MTPLX_MODEL,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temperature", "0.0",
        "--reasoning", "off",
        "--mtp",
        "--json",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    
    stdout = proc.stdout
    json_start = stdout.find("\n{\n")
    if json_start == -1:
        json_start = stdout.find("{\n")
    if json_start != -1:
        raw_json = stdout[json_start:].strip()
        try:
            data = json.loads(raw_json)
            stats = data.get("stats", {})
            return {
                "text": data.get("text", "").strip(),
                "reasoning": "",
                "elapsed_s": t1 - t0,
                "tok_s": stats.get("tok_s", 0.0),
                "completion_tokens": stats.get("generated_tokens", 0),
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "mtp_depth": stats.get("mtp_depth", 0),
                "accepted_by_depth": stats.get("accepted_by_depth", []),
            }
        except json.JSONDecodeError as e:
            return {"error": f"JSON decode failed: {e}", "text": stdout, "elapsed_s": t1 - t0, "tok_s": 0.0}
    return {"error": f"No JSON in output (code {proc.returncode}): {proc.stderr[:200]}", "text": "", "elapsed_s": t1 - t0, "tok_s": 0.0}

def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning quality and speed across models.")
    parser.add_argument("--models", nargs="+", default=["splash-q8", "splash-q4", "splash-mixed", "splash-hq", "mtplx-q8"],
                        choices=["splash-q8", "splash-q4", "splash-mixed", "splash-hq", "mtplx-q8"], help="Models to benchmark.")
    parser.add_argument("--port-hq", type=int, default=8003, help="Port for Splash-HQ (default: 8003).")
    parser.add_argument("--port-mixed", type=int, default=8004, help="Port for Splash-Mixed (default: 8004).")
    parser.add_argument("--limit", type=int, default=10, help="Number of questions per dataset (default: 10).")
    parser.add_argument("--max-tokens", type=int, default=400, help="Max tokens per generation.")
    parser.add_argument("--output", type=str, default="/tmp/quality_benchmark_results.json", help="Path to save results.")
    parser.add_argument("--force", action="store_true", help="Force rerun even if results exist.")
    args = parser.parse_args()

    print("=" * 80)
    print("AUTOMATED LLM QUALITY & SPEED BENCHMARK HARNESS")
    print(f"Models: {args.models} | Limit per dataset: {args.limit} | Output: {args.output}")
    print("=" * 80)

    # Load all benchmarks
    datasets = {
        "GPQA Diamond": load_gpqa(args.limit),
        "AIME 2025": load_aime25(args.limit),
        "MATH-500": load_math500(args.limit),
    }

    all_items = []
    for dname, ditems in datasets.items():
        print(f"Loaded {len(ditems)} items for {dname}")
        all_items.extend(ditems)

    # Load existing results if present
    results = {}
    if os.path.exists(args.output) and not args.force:
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded existing results from {args.output} ({len(results)} items)")
        except Exception:
            results = {}

    # Benchmark loops
    for model_key in args.models:
        print("\n" + "=" * 80)
        print(f"STARTING EVALUATION FOR: {model_key.upper()}")
        print("=" * 80)

        for i, item in enumerate(all_items, 1):
            qid = item["id"]
            if qid not in results:
                results[qid] = {
                    "benchmark": item["benchmark"],
                    "ground_truth": item["ground_truth"],
                    "type": item["type"],
                    "evaluations": {},
                }

            if model_key in results[qid]["evaluations"] and not args.force:
                existing = results[qid]["evaluations"][model_key]
                if "error" not in existing:
                    continue

            print(f"\n[{i}/{len(all_items)}] {item['benchmark']} - {qid} (GT: {item['ground_truth']})")
            
            # Execute model with appropriate token budget
            max_toks = 1500 if item["benchmark"] in ("AIME 2025", "MATH-500") else args.max_tokens
            if model_key == "splash-q8":
                res = run_splash_http(8001, "incoai/Qwen3.8-27B-Splash-Q8", item["prompt"], max_toks)
            elif model_key == "splash-q4":
                res = run_splash_http(8000, "incoai/Qwen3.8-27B-Splash", item["prompt"], max_toks)
            elif model_key == "splash-mixed":
                res = run_splash_http(args.port_mixed, "incoai/Qwen3.8-27B-Splash-Mixed", item["prompt"], max_toks)
            elif model_key == "splash-hq":
                res = run_splash_http(args.port_hq, "incoai/Qwen3.8-27B-Splash-HQ", item["prompt"], max_toks)
            elif model_key == "mtplx-q8":
                res = run_mtplx(item["prompt"], max_toks)
            else:
                res = {"error": f"Unknown model {model_key}"}

            if "error" in res:
                print(f"  [ERROR] {res['error']}")
                results[qid]["evaluations"][model_key] = {"error": res["error"]}
            else:
                pred = extract_answer(res.get("content") or "", item["type"])
                if pred is None:
                    pred = extract_answer(res.get("text") or "", item["type"])
                correct = grade_answer(pred, item["ground_truth"])
                mark = "CORRECT [PASS]" if correct else "INCORRECT [FAIL]"
                print(f"  Pred: {pred} | {mark} | Time: {res['elapsed_s']:.2f}s | Speed: {res['tok_s']:.1f} tok/s")
                results[qid]["evaluations"][model_key] = {
                    "prediction": pred,
                    "correct": correct,
                    "elapsed_s": res["elapsed_s"],
                    "tok_s": res["tok_s"],
                    "completion_tokens": res.get("completion_tokens", 0),
                    "text_snippet": res["text"][:150].strip() if res["text"] else "",
                }

            # Periodic checkpoint save
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    # Print Comparative Scorecard
    print_summary(results, args.models)

def print_summary(results, models):
    print("\n" + "=" * 90)
    print("FINAL BENCHMARK COMPARISON SCORECARD")
    print("=" * 90)

    benchmarks = ["GPQA Diamond", "AIME 2025", "MATH-500"]
    
    # Table header
    header = f"{'Benchmark':<16} |"
    for m in models:
        header += f" {m + ' Acc':<14} | {m + ' TPS':<12} |"
    print(header)
    print("-" * len(header))

    totals = {m: {"correct": 0, "total": 0, "tok_s": []} for m in models}

    for b in benchmarks:
        b_items = [v for v in results.values() if v.get("benchmark") == b]
        if not b_items:
            continue
        row = f"{b:<16} |"
        for m in models:
            evals = [v["evaluations"].get(m) for v in b_items if m in v.get("evaluations", {})]
            valid = [e for e in evals if e and "error" not in e]
            correct = sum(1 for e in valid if e.get("correct"))
            total = len(valid)
            acc = (correct / total * 100.0) if total > 0 else 0.0
            avg_tps = (sum(e.get("tok_s", 0) for e in valid) / len(valid)) if valid else 0.0
            row += f" {correct}/{total} ({acc:5.1f}%) | {avg_tps:6.1f} tok/s  |"
            totals[m]["correct"] += correct
            totals[m]["total"] += total
            totals[m]["tok_s"].extend([e.get("tok_s", 0) for e in valid if e.get("tok_s", 0) > 0])
        print(row)

    print("-" * len(header))
    total_row = f"{'OVERALL':<16} |"
    for m in models:
        c = totals[m]["correct"]
        t = totals[m]["total"]
        overall_acc = (c / t * 100.0) if t > 0 else 0.0
        overall_tps = (sum(totals[m]["tok_s"]) / len(totals[m]["tok_s"])) if totals[m]["tok_s"] else 0.0
        total_row += f" {c}/{t} ({overall_acc:5.1f}%) | {overall_tps:6.1f} tok/s  |"
    print(total_row)
    print("=" * 90)

if __name__ == "__main__":
    main()
