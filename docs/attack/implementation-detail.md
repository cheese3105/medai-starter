# Prompt Injection Implementation Detail

This document describes the technical implementation details of the prompt injection attack benchmark in the MED-AI framework.

---

## Table of Contents

- [1. Mathematical Formulation & Paper Mapping](#1-mathematical-formulation--paper-mapping)
  - [1.1 Task Formulation](#11-task-formulation)
  - [1.2 Mapping to MED-AI Pipeline](#12-mapping-to-med-ai-pipeline)
  - [1.3 Injection Payload Construction](#13-injection-payload-construction)
- [2. Deterministic Sampling & Pair Generation](#2-deterministic-sampling--pair-generation)
  - [2.1 Dataset Subsetting & Limits](#21-dataset-subsetting--limits)
  - [2.2 Cartesian Pairing & Seeded Shuffling](#22-cartesian-pairing--seeded-shuffling)
- [3. Architecture & Code Structure](#3-architecture--code-structure)
  - [3.1 Attack Registry & Task Adapters](#31-attack-registry--task-adapters)
  - [3.2 Execution Engine & Fault Tolerance](#32-execution-engine--fault-tolerance)
  - [3.3 Core Framework Integration](#33-core-framework-integration)
- [4. CLI Interface & Output Schema](#4-cli-interface--output-schema)
  - [4.1 Command-Line Interface](#41-command-line-interface)
  - [4.2 Incremental JSONL Data Files](#42-incremental-jsonl-data-files)
  - [4.3 Benchmark Summary Artifacts](#43-benchmark-summary-artifacts)

---

## 1. Mathematical Formulation & Paper Mapping

### 1.1 Task Formulation

The benchmark follows the indirect prompt injection task model established in adversarial literature. The primary clinical task is modeled as t = (s_t, x_t), where s_t denotes the system instructions for medical reasoning and x_t represents the medical case input. The injected task is modeled as e = (s_e, x_e), where s_e is an adversarial instruction to perform a secondary classification task and x_e is the injected data payload.

An attack function A(x_t, s_e, x_e) produces a compromised input x_tilde by embedding the injected instruction and data into the external source context. When the pipeline processes the input, the attacker seeks to hijack the model output such that the prediction matches the expected label of the injected task y_e rather than solving the legitimate clinical question.

### 1.2 Mapping to MED-AI Pipeline

Within MED-AI, the primary instruction s_t corresponds to the clinical prompt templates guiding the reasoning stage. The input x_t encompasses the patient vignette, multiple-choice options (A through D), and clean medical literature. The injected instruction s_e and payload x_e originate from standard NLP classification datasets, such as sentiment analysis or spam detection.

The compromised output x_tilde is passed into the pipeline through an external source parameter (`external_source`). This parameter is routed exclusively into the prompt template of the reasoning stage (`stages/reasoning.py`). In this manner, external unvalidated texts can be evaluated without contaminating internal vector retrieval collections or the state of the verification agent.

### 1.3 Injection Payload Construction

The benchmark provides five attack functions defined in `attack/attacks.py`, each constructing the compromised external source text differently:

- Naive: Directly concatenates the clean source, injected instruction, and injected data separated by single spaces.
- Escape Characters: Inserts double newline characters between the clean source and injected instruction, creating paragraph breaks that visually distinguish the new command from preceding medical text.
- Context Ignoring: Appends an explicit instruction override statement ("Ignore previous instructions.") immediately before the injected instruction.
- Fake Completion: Inserts a simulated task completion string ("Answer: task complete.") before presenting the injected instruction, encouraging the model to consider the prior task finished.
- Combined: Merges escape characters, fake completion text, and context ignoring into a single composite payload to maximize adversarial effectiveness.

---

## 2. Deterministic Sampling & Pair Generation

### 2.1 Dataset Subsetting & Limits

To balance comprehensive coverage with evaluation runtime, the CLI exposes independent limit parameters. The `--target-limit` flag selects the first N examples from the MedQA target dataset for baseline evaluation. The `--injected-limit` flag selects the first M examples from each injected classification dataset.

Separating target limits from injected limits allows the benchmark to establish stable clean baselines for both tasks before evaluating attacked pairs. The `--sample-size` flag then independently dictates how many attacked target-injected pairs are tested for each task.

### 2.2 Cartesian Pairing & Seeded Shuffling

Attacked evaluation pairs are generated via the `sample_pairs` function in `attack/sampling.py`. The function constructs the complete Cartesian product between the selected target questions and injected examples, producing all possible (target, injected) combinations.

To ensure deterministic evaluation across different runs, configurations, and architecture versions, the pair list is shuffled using an isolated `random.Random(seed)` generator initialized with `--seed` (default: 42). Slicing the first `--sample-size` pairs from this shuffled list guarantees that every architecture version (V0 through V3) and every attack method is tested against the exact same evaluation pairs.

---

## 3. Architecture & Code Structure

### 3.1 Attack Registry & Task Adapters

Attack methods are decoupled through a registry pattern in `attack/attacks.py`. Each attack function adheres to a uniform signature taking the clean source, injected instruction, and injected data string, and is mapped inside the `ATTACKS` registry dictionary. Adding a new attack requires only adding a function and registering its name.

Remote classification tasks are implemented in `attack/tasks.py`. The module downloads datasets from Hugging Face, caches them locally, and exposes uniform iterators across tasks. Each task adapter formats the raw sample into an instruction string, input data text, and target ground-truth label, and provides regex-based output parsing to extract predicted task labels from model answers.

### 3.2 Execution Engine & Fault Tolerance

The central execution coordinator is `attack/benchmark.py`. Execution proceeds in three distinct phases: evaluating clean MedQA target performance (`[PNA-T]`), evaluating clean injected task performance (`[PNA-I]`), and executing attacked episodes across all selected attack methods and tasks (`[ATTACK]`).

The execution loop wraps model calls with exception handling. If an individual request times out or encounters a remote provider error, the failure is logged and recorded in the output file without terminating the overall benchmark run. Furthermore, the underlying LLM client incorporates exponential backoff with jitter to handle temporary API rate limits automatically.

### 3.3 Core Framework Integration

The attack benchmark integrates cleanly with the main MED-AI core without modifying core pipeline logic. Data classes in `core/types.py` support an optional `external_source` field on `EpisodeInput` as well as raw response capture on `EpisodeResult`.

The runner controller in `core/runner.py` accepts this external source and injects it directly into the reasoning stage's prompt context. Other pipeline stages, such as ChromaDB vector retrieval in V1 or memory extraction in V3, continue operating without unintended interference from the injected payload.

---

## 4. CLI Interface & Output Schema

### 4.1 Command-Line Interface

The benchmark is invoked through `attack/cli.py` using Python's module execution syntax (`python -m attack.cli`). Command-line arguments configure the target configuration YAML (`--target-config`), attack methods (`--attacks`), injected tasks (`--tasks`), target dataset split (`--split`), sampling parameters (`--target-limit`, `--injected-limit`, `--sample-size`, `--seed`), and destination folder (`--output-dir`).

Default values are supplied for all parameters, allowing users to run quick smoke tests or extensive multi-task benchmarks using the same interface.

### 4.2 Incremental JSONL Data Files

To prevent data loss during lengthy evaluations, the benchmark streams episode results incrementally to three JSONL files in the output directory:

- `clean_targets.jsonl`: Contains baseline evaluation records for MedQA questions, recording target question IDs, ground-truth options, model predictions, latency, and token consumption.
- `clean_injected.jsonl`: Contains baseline evaluation records for injected classification tasks, recording dataset sample IDs, expected labels, model outputs, and execution metrics.
- `cases.jsonl`: Contains full attack episode records, documenting the target ID, injected task, attack method, whether the target answer remained correct, whether the injected task succeeded, latency, and token usage.

### 4.3 Benchmark Summary Artifacts

After all benchmark phases complete, the runner produces two final summary artifacts in the output directory:

- `metrics.json`: Contains aggregate calculations produced by `attack/metrics.py`. This includes PNA-T accuracy, per-task PNA-I accuracy, Attack Success Value (ASV), Matching Rate (MR), MedQA accuracy drop under attack, latency distributions (average, median, p95), and token totals.
- `run_config.json`: Records execution parameters, active model names, dataset splits, sampling limits, and timestamps. Sensitive values such as API keys are omitted to allow safe sharing and version-controlled experiment archiving.
