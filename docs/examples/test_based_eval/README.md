# TestBasedEval Demo

This directory contains simple examples showing how to use `TestBasedEval` for LLM evaluation.

## Two Approaches

### 1. Inline/Programmatic Usage

**File:** `inline_demo.py`

Create and run tests directly in Python code without files or CLI.

**Run it:**
```bash
python inline_demo.py
```

**Key features:**
- Create `TestBasedEval` instances inline
- Run evaluations programmatically
- Full control over the evaluation process

### 2. File-Based CLI Usage

**File:** `example_test.json`

Load tests from JSON files and run via CLI.

**Run it:**
```bash
m eval run example_test.json
```

**With custom models:**
```bash
m eval run example_test.json \
  --backend ollama \
  --model granite3.1-dense:2b \
  --judge-backend ollama \
  --judge-model granite3.1-dense:8b \
  --output-path results
```

**CLI Options:**
- `--backend, -b` - Generation backend (ollama, openai, hf, litellm)
- `--model` - Generation model name
- `--judge-backend, -jb` - Judge backend
- `--judge-model` - Judge model name
- `--output-path, -o` - Where to save results
- `--output-format` - json or jsonl (default: json)

## Test File Format

Test files are JSON with this structure:

```json
{
  "source": "benchmark_name",
  "name": "test_name",
  "instructions": "Evaluation criteria for the judge",
  "id": "unique_test_id",
  "examples": [
    {
      "input": [
        {"role": "user", "content": "prompt here"}
      ],
      "targets": [
        {"role": "assistant", "content": "expected response"}
      ],
      "input_id": "example_1"
    }
  ]
}
```

## Prerequisites

Make sure you have:
- Mellea installed
- Ollama running (or another backend configured)
- Required models downloaded (e.g., `ollama pull granite3.1-dense:2b`)
