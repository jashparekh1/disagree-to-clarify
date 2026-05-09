# D2C — Disagree to Clarify

**A conversational-AI approach to clarification as a grounding move.** When a user asks an ambiguous query, D2C treats the gap between interpretations as missing *common ground* between the user and the system. Three LLM agents with distinct interpretive lenses read the query independently, compare readings over multiple rounds, and — where their readings diverge — surface the divergence to a synthesizer, which produces a single clarifying question aimed at re-establishing common ground before any answer is attempted. For a full description of the system, experiments, and results, see the [technical report](report/d2c_final_report.pdf).

## Framing: clarification as grounding, not debate-to-consensus

Most multi-agent debate (MAD) work (e.g., Du et al. 2023) uses inter-agent disagreement as a *route to a better answer* — agents argue, one side wins, a consensus answer is emitted. D2C inverts this: the disagreement itself is the signal. If three reasonably-prompted agents read the same query differently, the query under-specifies what the system needs to act — i.e., the system lacks common ground with the user. The right conversational move in that situation is not to pick a winning interpretation; it is to ask a clarifying question so the user can ground the query themselves.

Concretely, D2C is a **clarification policy** over dialogue state:
- **Input:** a single user turn (possibly ambiguous).
- **Internal state:** each agent's interpretation, plus their round-over-round stance.
- **Output:** one conversational move — a clarifying question whose answer would add the specific common ground the interpretations lack.

Grounding in the Clark sense (Clark & Brennan 1991; Clark 1996) is the anchoring theory: a clarifying question is a grounding move, and D2C is an attempt to decide *which* grounding move to make using internal disagreement as the diagnostic.

## Setup

### Prerequisites

- Python >= 3.10
- [Ollama](https://ollama.com) installed locally **or** a vLLM server running on an OpenAI-compatible endpoint
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
```

### Backend: Ollama (default)

```bash
# Start Ollama
brew install ollama   # if not already installed
ollama serve

# Pull a model
ollama pull qwen3:4b
```

### Backend: vLLM (OpenAI-compatible)

```bash
# Start a vLLM server (example)
vllm serve Qwen/Qwen3-4B --port 8000

# Pass --backend openai and the HuggingFace model ID to all scripts
python -m scripts.run_demo "..." --backend openai --model Qwen/Qwen3-4B --no-think
```

> **Note on `--no-think`**: Qwen3 models default to thinking mode. When running on vLLM, pass `--no-think` to disable `<think>` token generation — otherwise agents output empty JSON. Ollama handles this automatically via its `think` field.

### Download evaluation datasets

```bash
python -m eval.datasets.download
```

This clones CLAMBER, Qulac, and ClariQ into `data/`.

## Quick Start

```bash
# Run D2C on a single query (defaults to the SAT-grounded variant, Ollama)
python -m scripts.run_demo "What is the best way to handle a Python crash?" --verbose

# Same query on vLLM
python -m scripts.run_demo "What is the best way to handle a Python crash?" --verbose \
  --backend openai --model Qwen/Qwen3-4B --no-think

# Run the pre-theory ablation (Literalist / Intent Seeker / Scope Expander)
python -m scripts.run_demo "How do I set up a table?" --verbose --variant original
```

## Usage

### Single query demo

```bash
python -m scripts.run_demo "Your ambiguous query here" [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `qwen3:4b` | Model name (Ollama tag or HuggingFace ID) |
| `--backend` | `ollama` | `ollama` or `openai` (vLLM / any OpenAI-compatible server) |
| `--base-url` | auto | Override server URL (default: `localhost:11434` for ollama, `localhost:8000` for openai) |
| `--rounds` | `3` | Number of dialogue rounds |
| `--max-tokens` | `2048` | Max tokens per LLM call |
| `--variant` | `speech_act` | Agent set: `speech_act` or `original` |
| `--no-think` | off | Disable thinking mode (recommended for vLLM + Qwen3) |
| `--verbose` | off | Print full round-by-round dialogue |

### Smoke test across datasets

```bash
# Run 3 examples from each of CLAMBER, ClariQ, Qulac — print traces + judge scores
python -m scripts.smoke_testsets

# With vLLM
python -m scripts.smoke_testsets --backend openai --model Qwen/Qwen3-4B --no-think

# More examples, specific dataset, skip judge
python -m scripts.smoke_testsets --n 10 --dataset clamber --no-judge
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n` | `3` | Examples per dataset |
| `--dataset` | `all` | `clariq`, `qulac`, `clamber`, or `all` |
| `--judge-model` | same as `--model` | Separate model for the LLM judge |
| `--no-judge` | off | Skip judge scoring, just print traces |
| `--rounds` | `3` | Dialogue rounds |
| `--backend` / `--base-url` / `--no-think` | — | Same as run_demo |

### Batch run

```bash
python -m scripts.run_batch --input data/queries.jsonl --output outputs/results.jsonl
```

Input JSONL format: one `{"query": "..."}` per line. Supports `--resume` to skip already-processed queries and `--max-workers` for parallel execution.

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

# Analyze results
python -m eval.analyze --results-dir results/
```

### Baselines

```bash
# Run vanilla CQG and parallel baselines
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

### Dialogue mechanics

Each agent maintains its own **multi-turn message history** across rounds:

```
[system: role definition]
[user:   original query]          ← round 0
[asst:   interpretation]
[user:   other agents' round-0 readings + decision tree]   ← round 1
[asst:   interpretation + stance + reason]
[user:   other agents' round-1 readings + decision tree]   ← round 2
[asst:   interpretation + stance + reason]
...
```

In rounds 1+, each agent sees only the **previous round's** readings from other **active** agents (already-conceded agents are listed as dropped, not shown). This means each agent's context grows naturally with the conversation without carrying redundant cross-agent history.

**Stance mechanics:**

| Stance | When to use |
|--------|-------------|
| `HOLD` | You can name a specific gap no other agent has captured (default) |
| `UPDATE` | Others have partially shifted your view but not fully covered your concern |
| `CONCEDE` | Another agent's reading covers your exact concern — name who and why |

Conceded agents drop out of subsequent rounds (sticky). The dialogue stops early if all active agents concede in the same round. The synthesizer always receives the full transcript regardless of early stopping, but focuses on HOLD/UPDATE agents' final stances as its primary signal.

### Agent variants

Each variant gives the three agents a different *kind* of reading to hold. They are not "sides of an argument"; they are lenses that, when they diverge, tell us *what kind of common ground is missing*.

**Speech Act Theory — the default, SAT-grounded trio.** Grounded in Austin (1962) and Searle (1969, 1975, 1976). The three lenses map one-to-one onto Austin's tripartite decomposition of an utterance, and divergence between them maps cleanly onto three distinct grounding gaps:

- **Locutionary Parser** — Austin's phatic + rhetic acts. Attends only to form (syntax) and sense/reference (lexical semantics, referent fixing). Divergence here ⇒ **referential grounding gap**.
- **Illocutionary Analyst** — Searle 1976's five-way classification (Assertive / Directive / Commissive / Expressive / Declaration) plus Searle 1975 on indirect speech acts. Attends to which act the user is *performing* with the query. Divergence ⇒ **intent grounding gap**.
- **Perlocutionary Evaluator** — Austin's perlocutionary act: the effect the utterance is intended to produce in the hearer, distinct from felicity conditions. Attends to the situated context required for the response to actually land. Divergence ⇒ **pragmatic grounding gap**.

**Original D2C** (`--variant original`, kept for ablation). An atheoretical trio used in early experiments and retained so we can compare SAT-grounded against pre-theory decomposition:
- **Literalist** — surface-level, dictionary-default reading.
- **Intent Seeker** — infers the user's underlying goal.
- **Scope Expander** — identifies what the query leaves unspecified.

### References

- Austin, J. L. (1962). *How to Do Things with Words*. Oxford: Clarendon Press.
- Clark, H. H., & Brennan, S. E. (1991). "Grounding in Communication." In L. B. Resnick, J. M. Levine, & S. D. Teasley (eds.), *Perspectives on Socially Shared Cognition*. APA Books.
- Searle, J. R. (1969). *Speech Acts: An Essay in the Philosophy of Language*. Cambridge University Press.
- Searle, J. R. (1975). "Indirect Speech Acts." In P. Cole & J. L. Morgan (eds.), *Syntax and Semantics 3: Speech Acts*. Academic Press.
- Searle, J. R. (1976). "A Classification of Illocutionary Acts." *Language in Society*, 5(1), 1–23.

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
│   ├── llm.py              # Ollama + OpenAI-compatible (vLLM) HTTP client
│   ├── prompts.py          # All prompt templates
│   ├── agents.py           # Agent roles, stance mechanics, per-agent message history
│   ├── dialogue.py         # Multi-round dialogue loop with sticky CONCEDE
│   ├── synthesizer.py      # Dialogue transcript → clarifying question
│   ├── pipeline.py         # End-to-end orchestration
│   └── baseline.py         # Vanilla CQG baseline
├── eval/
│   ├── datasets/           # Unified loaders for CLAMBER, Qulac, ClariQ
│   ├── judge.py            # Binary LLM-as-judge (match / no-match)
│   ├── judge_prompts.py    # Judge prompt templates
│   ├── metrics.py          # F1, semantic similarity
│   ├── run_eval.py         # Evaluation runner CLI
│   └── analyze.py          # Results tables and analysis
├── scripts/
│   ├── run_demo.py         # Single query demo
│   ├── smoke_testsets.py   # N examples × dataset, traces + judge scores
│   ├── run_batch.py        # Batch processing
│   ├── run_baselines.py    # Vanilla + parallel baselines
│   ├── run_rl_baseline.py  # RL baseline
│   ├── inspect_data.py     # Dataset inspection
│   ├── dataset_stats.py    # Dataset statistics
│   └── build_test_sets.py  # Build JSONL test sets from raw datasets
├── test_sets/              # Pre-built JSONL test sets (clariq, qulac, clamber)
├── data/                   # Raw datasets (downloaded separately)
├── outputs/                # D2C outputs (JSONL)
└── results/                # Evaluation results
```

## Notes

- **Model choice**: `qwen3:4b` is the minimum recommended size.
- **Thinking tags**: Qwen3 models produce `<think>...</think>` blocks which are stripped automatically. On vLLM, pass `--no-think` to disable thinking entirely — without it, models put reasoning inside `<think>` tags and emit empty JSON content.
- **Token budget**: Default `max_tokens=2048` accounts for thinking token overhead. Lower values may produce empty responses with reasoning models.
- **Structured output**: Ollama enforces JSON schemas natively via the `format` field. vLLM uses `response_format: {type: "json_object"}` which enforces valid JSON but not field-level constraints — prompt-level hints in every user turn compensate for this.
