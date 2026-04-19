# D2C — Disagree to Clarify

Multi-agent dialogue system for query disambiguation. Three LLM agents with distinct interpretive lenses debate an ambiguous query over multiple rounds, then a synthesizer identifies the key disagreement and produces a targeted clarifying question.

## Setup

### Prerequisites

- Python >= 3.10
- [Ollama](https://ollama.com) installed locally
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/disagree-to-clarify.git
cd disagree-to-clarify

# 2. Create a virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Start Ollama (in a separate terminal)
brew install ollama   # if not already installed
ollama serve

# 4. Pull a model
ollama pull qwen3:4b
```

### Download evaluation datasets

```bash
python -m eval.datasets.download
```

This clones CLAMBER, Qulac, and ClariQ into `data/`.

## Quick Start

```bash
# Run D2C on a single query
python -m scripts.run_demo "What is the best way to handle a Python crash?" --verbose

# Run with Speech Act Theory agents
python -m scripts.run_demo "How do I set up a table?" --verbose --variant speech_act
```

## Usage

### Single query demo

```bash
python -m scripts.run_demo "Your ambiguous query here" [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `qwen3:4b` | Ollama model name |
| `--rounds` | `3` | Number of dialogue rounds |
| `--max-tokens` | `2048` | Max tokens per LLM call |
| `--verbose` | off | Print full round-by-round dialogue |

### Batch run

```bash
python -m scripts.run_batch --input data/queries.jsonl --output outputs/results.jsonl
```

Input JSONL format: one `{"query": "..."}` per line. Supports `--resume` to skip already-processed queries and `--max-workers` for parallel execution.

### ClarifyMT-Bench evaluation

```bash
# Run on a small sample
python -m scripts.run_clarifymt --limit 10 --output outputs/clarifymt_results.jsonl

# Run with LLM-as-judge scoring
python -m scripts.run_clarifymt --limit 10 --judge --output outputs/clarifymt_scored.jsonl

# Evaluate existing results
python -m scripts.run_clarifymt --eval-only --input outputs/clarifymt_scored.jsonl
```

### Full evaluation suite (CLAMBER, Qulac, ClariQ)

```bash
# Download datasets first
python -m eval.datasets.download

# Inspect dataset examples
python -m scripts.inspect_data --dataset clamber --n 5

# Print dataset statistics
python -m scripts.dataset_stats

# Run evaluation
python -m eval.run_eval --dataset clamber --input outputs/d2c_clamber.jsonl --judge-model qwen3:8b

# Run on all datasets
python -m eval.run_eval --dataset all --input-dir outputs/ --skip-judge

# Analyze results
python -m eval.analyze --results-dir results/
```

### Baselines

```bash
# Run vanilla and parallel baselines
python -m scripts.run_baselines --input data/ambigqa_sample.jsonl --output outputs/baselines.jsonl

# Run RL baseline
python -m scripts.run_rl_baseline --input data/ambigqa_sample.jsonl --output outputs/rl_baseline.jsonl
```

## Architecture

```
                        ┌─────────────┐
                        │   Query     │
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              v                v                v
        ┌───────────┐  ┌─────────────┐  ┌──────────────┐
        │ Agent 1   │  │  Agent 2    │  │   Agent 3    │
        └─────┬─────┘  └──────┬──────┘  └──────┬───────┘
              │                │                │
              └────────┬───────┴────────┬───────┘
                       │  N rounds of   │
                       │    debate      │
                       └───────┬────────┘
                               │
                       ┌───────v────────┐
                       │  Synthesizer   │
                       └───────┬────────┘
                               │
                    ┌──────────v──────────┐
                    │ Clarifying Question │
                    └─────────────────────┘
```

### Agent Variants

**Original D2C** (default):
- **Literalist** — surface-level, dictionary-default reading
- **Intent Seeker** — infers the user's underlying goal
- **Scope Expander** — identifies what the query leaves unspecified

**Speech Act Theory** (`--variant speech_act`):
- **Locutionary Parser** — analyzes the physical utterance (words, grammar)
- **Illocutionary Analyst** — identifies the intended action/force behind the query
- **Perlocutionary Evaluator** — considers what context is needed for a successful effect

### Evaluation Metrics

| Metric | Description | Datasets |
|--------|-------------|----------|
| Clarification Need F1 | Binary: should the system ask a clarifying question? | CLAMBER, ClariQ |
| Judge Quality (1-5) | LLM-as-judge scores generated vs gold clarifying question | All |
| Semantic Similarity | Embedding cosine similarity (all-MiniLM-L6-v2) | All |

For Qulac and ClariQ, metrics use max-over-facets scoring (multiple valid clarifying questions per topic).

## Project Structure

```
d2c/
├── d2c/
│   ├── llm.py              # Ollama HTTP client
│   ├── prompts.py           # All prompt templates
│   ├── agents.py            # Agent roles and response parsing
│   ├── dialogue.py          # Multi-round dialogue loop
│   ├── synthesizer.py       # Dialogue → clarifying question
│   ├── pipeline.py          # End-to-end orchestration
│   └── data.py              # ClarifyMT-Bench data loader
├── eval/
│   ├── datasets/            # Unified loaders for CLAMBER, Qulac, ClariQ
│   ├── metrics.py           # F1, semantic similarity, LLM judge
│   ├── judge_prompts.py     # LLM-as-judge prompt templates
│   ├── run_eval.py          # Evaluation runner CLI
│   └── analyze.py           # Results tables and analysis
├── scripts/
│   ├── run_demo.py          # Single query demo
│   ├── run_batch.py         # Batch processing
│   ├── run_clarifymt.py     # ClarifyMT-Bench runner
│   ├── run_baselines.py     # Vanilla + parallel baselines
│   ├── run_rl_baseline.py   # RL baseline
│   ├── inspect_data.py      # Dataset inspection
│   └── dataset_stats.py     # Dataset statistics
├── data/                    # Datasets (downloaded separately)
├── outputs/                 # D2C outputs (JSONL)
└── results/                 # Evaluation results
```

## Notes

- **Model choice**: `qwen3:4b` is the default. Larger models (e.g., `qwen3:8b`) produce more consistent structured output. Use a different model for the judge than for agents.
- **Thinking tags**: qwen3 models produce `<think>...</think>` blocks which are stripped automatically.
- **Token budget**: Default `max_tokens=2048` accounts for thinking token overhead. Lower values may produce empty responses with reasoning models.
