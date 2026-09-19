# Quickstart Guide

This guide provides instructions to set up the environment, run automated benchmarks, launch interactive chat sessions, and execute evaluation and security testing within the MED-AI framework.

---

## Table of Contents

- [1. Environment Setup & Installation](#1-environment-setup--installation)
  - [1.1 Prerequisites & Python Environment](#11-prerequisites--python-environment)
  - [1.2 Installing Dependencies](#12-installing-dependencies)
  - [1.3 Configuring Environment Variables](#13-configuring-environment-variables)
- [2. Running & Evaluating MED-AI](#2-running--evaluating-med-ai)
  - [2.1 Automated Benchmark Execution](#21-automated-benchmark-execution)
  - [2.2 Interactive Chat REPL](#22-interactive-chat-repl)
  - [2.3 Evaluation & Performance Metrics](#23-evaluation--performance-metrics)
- [3. Running Prompt Injection Attacks](#3-running-prompt-injection-attacks)
  - [3.1 Smoke Testing the Attack Runner](#31-smoke-testing-the-attack-runner)
  - [3.2 Full Attack Execution](#32-full-attack-execution)
  - [3.3 Attack Benchmark Procedure](#33-attack-benchmark-procedure)

---

## 1. Environment Setup & Installation

### 1.1 Prerequisites & Python Environment

MED-AI requires Python 3.10 or higher.

Create a virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 1.2 Installing Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### 1.3 Configuring Environment Variables

Copy the example environment file and configure your credentials:

```bash
cp .env.example .env
```

Open `.env` and set the endpoint, model name, and API keys for your provider:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

Local inference servers such as Ollama or vLLM can also be configured by updating `OPENAI_BASE_URL` to point to the local server URL.

---

## 2. Running & Evaluating MED-AI

### 2.1 Automated Benchmark Execution

Run the MedQA-USMLE automated benchmark via `main.py`:

```bash
python main.py --mode benchmark --config <config-path> [options]
```

Parameters and options:
- `--config`: Path to the YAML configuration file selecting the architecture version:
  - `configs/v0.yaml`: Baseline direct LLM reasoning.
  - `configs/v1.yaml`: RAG retrieval with ChromaDB evidence.
  - `configs/v2.yaml` or `configs/v2-qr.yaml`: Self-correction with query rewriting.
  - `configs/v3.yaml` or `configs/v3-qr.yaml`: Long-term memory-augmented reasoning.
- `--split`: Target dataset split (`test` by default, or `train`).
- `--limit`: Maximum number of questions to evaluate (e.g. `--limit 20`).
- `--workers`: Number of parallel worker threads (default: 8).
- `--output`: Custom output filepath for the predictions JSONL file.

Example:

```bash
python main.py --mode benchmark --config configs/v0.yaml --limit 20 --workers 4 --output output/v0_test_20.jsonl
```

### 2.2 Interactive Chat REPL

Launch a multi-turn clinical chat session with memory support:

```bash
python main.py --mode chat --config configs/v3-chat.yaml
```

Parameters and commands:
- `--config`: Path to the chat configuration file (`configs/v3-chat.yaml` enables short-term session buffering and long-term memory retrieval).
- REPL commands during chat:
  - Type `exit` or `quit` to end the session.
  - Type `clear` to reset the active short-term conversation buffer.

### 2.3 Evaluation & Performance Metrics

Evaluate generated predictions and compare versions using `evaluate/evaluate.py`:

```bash
python evaluate/evaluate.py [options]
```

Parameters and options:
- `--files <file1> <file2> ...`: One or more prediction JSONL files to evaluate.
- `--dir <directory>`: Directory containing prediction files to batch evaluate.
- `--baseline <prefix>`: Baseline identifier (e.g., `v0`) to perform pairwise statistical comparisons against.
- `--report <path.md>`: Optional output file to generate a Markdown evaluation report.

Example:

```bash
python evaluate/evaluate.py --dir output/ --baseline v0 --report output/report.md
```

Outputs:
- Computes accuracy, invalid prediction rates, latency, and estimated token costs.
- Runs McNemar's Test to verify statistical significance ($p < 0.05$) between versions.
- Calculates Bootstrap 95% Confidence Intervals for accuracy and win/loss differentials.
- Saves `summary_metrics.csv` and `pairwise_comparison.csv`.

---

## 3. Running Prompt Injection Attacks

### 3.1 Smoke Testing the Attack Runner

Run a quick sanity check with reduced samples to ensure the attack pipeline functions:

```bash
.venv/bin/python -m attack.cli \
  --target-config attack/configs/v0-attack.yaml \
  --attacks naive escape_characters \
  --tasks sentiment \
  --target-limit 2 \
  --injected-limit 2 \
  --sample-size 2 \
  --seed 42 \
  --output-dir attack/output/smoke-test
```

### 3.2 Full Attack Execution

Run prompt injection benchmarks across attack types and injected tasks:

```bash
.venv/bin/python -m attack.cli --target-config <config-path> [options]
```

Parameters and options:
- `--target-config`: Attack configuration overlay (`attack/configs/v0-attack.yaml`, `v1-qr-attack.yaml`, `v2-qr-attack.yaml`, `v3-qr-attack.yaml`).
- `--attacks`: Attack methods to evaluate (`naive`, `escape_characters`, `context_ignoring`, `fake_completion`, `combined`).
- `--tasks`: Injected classification tasks (`sentiment`, `spam`, `duplicate`, `hate`, `nli`).
- `--target-limit`: Number of MedQA target questions.
- `--injected-limit`: Number of injected examples per task.
- `--sample-size`: Number of target/injected pairs evaluated per task.
- `--seed`: Random seed for deterministic pairing (default: 42).
- `--output-dir`: Output directory for JSONL logs and `metrics.json`.

### 3.3 Attack Benchmark Procedure

When the benchmark is launched, it executes in three sequential phases:

1. Clean Target Baseline (`[PNA-T]`): Runs the selected MedQA questions through the pipeline without attack payloads to record baseline medical reasoning accuracy and latency.
2. Clean Injected Baseline (`[PNA-I]`): Prompts the model directly with clean injected task examples (e.g., sentiment or spam classification) to establish baseline task competency.
3. Attacked Evaluation (`[ATTACK]`): Forms deterministic pairs between target questions and injected examples using a fixed random seed. The chosen attack method embeds the malicious instruction into the external source, which is then fed into the pipeline's reasoning stage.

Results are written incrementally to `clean_targets.jsonl`, `clean_injected.jsonl`, and `cases.jsonl`. Once complete, aggregate metrics including Attack Success Value (ASV), Matching Rate (MR), latency statistics, and token counts are computed and saved to `metrics.json`.
