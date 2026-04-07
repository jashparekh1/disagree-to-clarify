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

## Architecture

```
Query → [Literalist, Intent Seeker, Scope Expander] → N rounds of debate → Synthesizer → Clarifying Question
```

Each agent interprets the query through a different lens and defends its perspective across rounds. The synthesizer identifies the most consequential disagreement and generates a single clarifying question.
