# D2C — Disagree to Clarify

Multi-agent dialogue system for query disambiguation. Three LLM agents with distinct interpretive lenses debate an ambiguous query, then a synthesizer produces a targeted clarifying question.

## Setup

1. Install [Ollama](https://ollama.com) and start the server:
   ```bash
   ollama serve
   ```

2. Pull a model:
   ```bash
   ollama pull qwen3:4b
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Single query demo
```bash
python -m scripts.run_demo "What is the best way to deal with a Python crash?" --verbose
```

Flags:
- `--model` — Ollama model name (default: `qwen3:4b`)
- `--rounds` — number of dialogue rounds (default: `3`)
- `--verbose` — print full round-by-round dialogue

### Batch run
```bash
python -m scripts.run_batch --input data/queries.jsonl --output outputs/results.jsonl
```

Input JSONL format: one `{"query": "..."}` per line. Supports `--resume` to skip already-processed queries.

## Evaluation

### Metrics

| Metric | What it measures | Dataset |
|---|---|---|
| **Interpretation Recall** | Fraction of gold interpretations covered by at least one agent interpretation | AmbigQA |
| **Interpretation Precision** | Fraction of agent interpretations that match a valid gold interpretation | AmbigQA |
| **CQ Similarity** | Semantic similarity between generated and gold clarifying question | ClarifyMT-Bench |
| **LLM Judge Score** | LLM-as-judge (1–5) on whether the question resolves the key ambiguity | ClarifyMT-Bench |
| **ROUGE-L** | Lexical overlap between generated and reference answer | AmbigQA / ASQA |
| **Disambiguation F1** | Token-level F1 averaged over gold answers per interpretation | AmbigQA / ASQA |
| **MRR / nDCG@20** | Retrieval quality improvement with clarified query vs. original | ClariQ |

Semantic similarity uses `all-MiniLM-L6-v2` via `sentence-transformers`. Recall/precision use a threshold (default `0.6`) calibrated on a dev set.

### Single query

Runs the full D2C pipeline and scores the output immediately:

```bash
python -m scripts.run_eval single \
  --query "How do I deal with a Python crash?" \
  --gold-interpretations "fixing a runtime error" "handling a syntax error" \
  --gold-cq "Are you asking about a runtime crash or a syntax error?" \
  --model qwen2.5:0.5b \
  --threshold 0.3
```

Add `--llm-judge` to also run the LLM judge (makes extra LLM calls).

### Batch evaluation

First run D2C on your dataset:

```bash
python -m scripts.run_batch --input data/queries.jsonl --output outputs/results.jsonl
```

Then score against a gold file:

```bash
python -m scripts.run_eval batch \
  --results outputs/results.jsonl \
  --gold data/gold.jsonl \
  --output outputs/eval_scores.jsonl
```

Gold JSONL format — one JSON object per line:

```json
{"query": "...", "gold_interpretations": ["...", "..."], "gold_cq": "...", "gold_answers": ["...", "..."]}
```

`gold_cq` and `gold_answers` are optional. The script prints aggregate scores and writes per-query scores to the output file.

## Architecture

```
Query → [Literalist, Intent Seeker, Scope Expander] → N rounds of debate → Synthesizer → Clarifying Question
```

Each agent interprets the query through a different lens and defends its perspective across rounds. The synthesizer identifies the most consequential disagreement and generates a single clarifying question.
