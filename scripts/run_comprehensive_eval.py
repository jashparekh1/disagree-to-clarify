"""Comprehensive Evaluation Master Script.

Workflow:
1. Ensure Data: Download AmbigQA if missing.
2. Filter Data: Create 100-item evaluation set.
3. SFT Setup: Generate SFT dataset and train LoRA if adapters missing.
4. Run Evals:
   - Vanilla Baseline
   - Parallel Baseline
   - SFT Baseline (MLX)
   - D2C (Dialogue) - Your System
5. Report: Generate Summary Table and Plot.
"""

import os
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import logging
import argparse

# Configuration
EVAL_N = 100  # Full robust run
MODEL_OLLAMA = "qwen2.5:3b"
MODEL_JUDGE = "llama3.1:8b"
MODEL_CUDA_BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "adapters/"
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
SFT_DATA_DIR = os.path.join(DATA_DIR, "sft")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def run_cmd(cmd, desc):
    logger.info(f"--- {desc} ---")
    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        logger.error(f"Command failed with code {result.returncode}")
        # Not raising exit so we can try to continue other parts if one fails
    return result.returncode

def ensure_infrastructure():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Download AmbigQA if missing
    if not os.path.exists(os.path.join(DATA_DIR, "dev_light.json")):
        run_cmd(
            f"curl -L https://nlp.cs.washington.edu/ambigqa/data/ambignq_light.zip -o {DATA_DIR}/ambignq_light.zip && unzip -o {DATA_DIR}/ambignq_light.zip -d {DATA_DIR}/",
            "Downloading AmbigQA dataset"
        )

    # 2. Create Filtered Eval Set (100 items)
    eval_set_path = os.path.join(DATA_DIR, f"eval_master_{EVAL_N}.jsonl")
    if not os.path.exists(eval_set_path):
        logger.info(f"Creating evaluation set with {EVAL_N} items...")
        with open(os.path.join(DATA_DIR, "dev_light.json")) as f:
            data = json.load(f)
        
        filtered = []
        for item in data:
            ints = []
            for ann in item.get("annotations", []):
                if ann.get("type") == "multipleQAs":
                    ints.extend([qa["question"] for qa in ann.get("qaPairs", []) if "question" in qa])
            if len(ints) >= 2:
                filtered.append(item)
            if len(filtered) >= EVAL_N:
                break
        
        with open(eval_set_path, "w") as out:
            for item in filtered:
                out.write(json.dumps(item) + "\n")

    # 3. SFT Dataset and Training
    if not os.path.exists(ADAPTER_PATH):
        logger.info("Adapters not found. Starting SFT pipeline...")
        # Dataset
        run_cmd("PYTHONPATH=. python3 scripts/prepare_sft_dataset.py", "Preparing SFT Dataset")
        # Training (CUDA)
        run_cmd(
            f"PYTHONPATH=. python3 trainers/cuda/sft_train.py",
            "Training SFT Model (CUDA)"
        )

def run_evaluations(max_workers: int, backend: str = "ollama", base_url: str | None = None):
    eval_input = os.path.join(DATA_DIR, f"eval_master_{EVAL_N}.jsonl")
    
    extra_args = f"--backend {backend}"
    if base_url:
        extra_args += f" --base-url {base_url}"

    # A. Vanilla & Parallel
    run_cmd(
        f"PYTHONPATH=. python3 scripts/run_baselines.py --input {eval_input} --output-prefix {OUTPUT_DIR}/master_baselines --model {MODEL_OLLAMA} --judge-model {MODEL_JUDGE} --max-workers {max_workers} {extra_args}",
        "Running Vanilla and Parallel Baselines"
    )
    
    # B. D2C (Your System)
    run_cmd(
        f"PYTHONPATH=. python3 scripts/run_eval_ambigqa.py --input {eval_input} --output {OUTPUT_DIR}/master_d2c.jsonl --model {MODEL_OLLAMA} --judge-model {MODEL_JUDGE} --variant original --max-workers {max_workers} {extra_args}",
        "Running D2C (Dialogue) System - Original"
    )

    # B2. D2C (Speech Act)
    run_cmd(
        f"PYTHONPATH=. python3 scripts/run_eval_ambigqa.py --input {eval_input} --output {OUTPUT_DIR}/master_d2c_speech_act.jsonl --model {MODEL_OLLAMA} --judge-model {MODEL_JUDGE} --variant speech_act --max-workers {max_workers} {extra_args}",
        "Running D2C (Dialogue) System - Speech Act"
    )
    
    # C. SFT (CUDA)
    run_cmd(
        f"PYTHONPATH=. python3 scripts/run_eval_sft_cuda.py --input {eval_input} --output {OUTPUT_DIR}/master_sft.jsonl --model {ADAPTER_PATH} --judge-model {MODEL_OLLAMA}",
        "Running SFT-Tuned Baseline (CUDA)"
    )


def aggregate_and_report():
    logger.info("Aggregating results...")
    metrics = []
    
    files = {
        "Vanilla": os.path.join(OUTPUT_DIR, "master_baselines_vanilla.jsonl"),
        "Parallel": os.path.join(OUTPUT_DIR, "master_baselines_parallel.jsonl"),
        "D2C (Original)": os.path.join(OUTPUT_DIR, "master_d2c.jsonl"),
        "D2C (Speech Act)": os.path.join(OUTPUT_DIR, "master_d2c_speech_act.jsonl"),
        "SFT (LoRA)": os.path.join(OUTPUT_DIR, "master_sft.jsonl"),
    }
    
    for name, path in files.items():
        if not os.path.exists(path):
            logger.warning(f"Result file missing for {name}: {path}")
            continue
            
        scores = []
        coverage = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                if "eval" in data:
                    scores.append(data["eval"].get("score", 0))
                    coverage.append(1 if data["eval"].get("covers_interpretations") else 0)
        
        if scores:
            metrics.append({
                "System": name,
                "Avg Quality Score": sum(scores) / len(scores),
                "Coverage %": (sum(coverage) / len(coverage)) * 100
            })
    
    df = pd.DataFrame(metrics)
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    # Save table
    df.to_csv(os.path.join(OUTPUT_DIR, "comparison_results.csv"), index=False)
    
    # Generate Plot
    plt.figure(figsize=(12, 6))
    plt.bar(df["System"], df["Avg Quality Score"], color=['gray', 'blue', 'green', 'cyan', 'orange'])
    plt.title("Comparison of Clarifying Question Quality (LLM-as-Judge)")
    plt.ylabel("Avg Score (1-5)")
    plt.ylim(0, 5)
    for i, score in enumerate(df["Avg Quality Score"]):
        plt.text(i, score + 0.1, f"{score:.2f}", ha='center', fontweight='bold')
    
    plot_path = os.path.join(OUTPUT_DIR, "comparison_plot.png")
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel workers for LLM calls")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "openai"],
                        help="LLM backend: ollama (default) or openai (vLLM / any OpenAI-compatible server)")
    parser.add_argument("--base-url", default=None,
                        help="Override LLM server URL (default: localhost:11434 for ollama, localhost:8000 for openai)")
    args = parser.parse_args()

    ensure_infrastructure()
    run_evaluations(args.max_workers, backend=args.backend, base_url=args.base_url)
    aggregate_and_report()

if __name__ == "__main__":
    main()

