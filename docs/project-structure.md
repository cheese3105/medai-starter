# Project Structure

This document describes the directory structure, architectural layers, core modules, and subsystems of the MED-AI framework.

---

## Table of Contents

- [1. Architecture Progression (V0 to V3)](#1-architecture-progression-v0-to-v3)
  - [1.1 Architectural Evolution](#11-architectural-evolution)
  - [1.2 Modular Pipeline Principles](#12-modular-pipeline-principles)
- [2. Core Runtime & Pipeline Stages](#2-core-runtime--pipeline-stages)
  - [2.1 Core Runtime Engine (`core/`)](#21-core-runtime-engine-core)
  - [2.2 Pipeline Execution Stages (`stages/`)](#22-pipeline-execution-stages-stages)
  - [2.3 Memory Management (`memory/`)](#23-memory-management-memory)
- [3. Operational Directories & Resources](#3-operational-directories--resources)
  - [3.1 Configuration Management (`configs/`)](#31-configuration-management-configs)
  - [3.2 Data Stores & Vector Indices (`data/`)](#32-data-stores--vector-indices-data)
  - [3.3 Artifacts & Output Tracking (`output/`, `evaluate/`)](#33-artifacts--output-tracking-output-evaluate)
- [4. Execution Modes & Security Benchmark](#4-execution-modes--security-benchmark)
  - [4.1 Entrypoints & Operational Modes (`modes/`, `main.py`)](#41-entrypoints--operational-modes-modes-mainpy)
  - [4.2 Attack Benchmark Integration (`attack/`)](#42-attack-benchmark-integration-attack)

---

## 1. Architecture Progression (V0 to V3)

MED-AI is designed as an evolving clinical reasoning system. It progresses across four architectures to benchmark how retrieval, self-correction, and memory impact diagnostic accuracy and reliability.

### 1.1 Architectural Evolution

The framework supports four distinct architecture versions:

- V0 (Direct LLM Reasoning): A baseline zero-shot or few-shot reasoning setup. The LLM receives the patient query directly without external knowledge retrieval, verification agents, or memory.
- V1 (RAG Evidence Retrieval): Augments the reasoning model with Retrieval-Augmented Generation. Medical textbook passages are retrieved from a ChromaDB vector database and supplied as clinical evidence before generating an answer.
- V2 (Self-Correction & Query Rewriting): Adds an iterative verification loop. A Verifier agent examines whether the draft answer is supported by the retrieved evidence. If the evidence is insufficient, a Query Rewriter agent refines the query to fetch new passages, looping until supported or reaching the iteration limit.
- V3 (Memory-Augmented System): Combines the V2 loop with a dual-layer memory system. A Short-Term Memory buffer maintains conversational flow across turns, while a Long-Term Memory module extracts and persists patient-specific facts (allergies, medical history, medications) in ChromaDB.

### 1.2 Modular Pipeline Principles

The system follows three key architectural rules:

- Stage Decoupling: Each processing step (retrieval, reasoning, verification, rewriting) is implemented as an independent stage conforming to a shared interface.
- Structured State Passing: Data flows between stages through strongly typed objects (`EpisodeInput`, `StageOutput`, and `EpisodeResult`).
- Configuration-Driven Execution: The entire pipeline topology, model parameters, iteration limits, and active stages are controlled by YAML configuration files, allowing seamless switching between versions without changing code.

---

## 2. Core Runtime & Pipeline Stages

The execution engine is separated into low-level runtime controllers and specialized task stages.

### 2.1 Core Runtime Engine (`core/`)

The `core/` package provides runtime control, data schemas, logging, and client management:

- `core/runner.py`: The pipeline coordinator. It executes stages in the configured sequence, manages the verification retry loop, tracks token limits, and handles early stopping when a supported answer is achieved.
- `core/llm_client.py`: Provides unified access to LLM providers (LangChain OpenAI/Ollama clients). It includes rate-limit handling, exponential backoff, and retry logic to ensure reliable execution during long benchmark runs.
- `core/config.py`: Reads YAML configuration files, resolves environment variables from `.env`, and validates runtime settings.
- `core/types.py`: Defines the primary dataclasses and TypedDicts used across the application, such as `EpisodeInput`, `EpisodeResult`, and `StageOutput`.
- `core/logger.py`: Handles structured JSONL event logging and stdout progress reporting for running episodes.

### 2.2 Pipeline Execution Stages (`stages/`)

The `stages/` package contains individual pipeline components that inherit from a common interface:

- `stages/base.py`: Defines the `BaseStage` abstract base class with a standard `execute()` contract.
- `stages/retrieval.py`: Handles vector search against ChromaDB, taking the current query and returning the most relevant textbook chunks.
- `stages/reasoning.py`: Generates the clinical reasoning and candidate answers using prompt templates populated with the question, options, retrieved evidence, and optional external data.
- `stages/verifier.py`: Analyzes the candidate answer against the retrieved evidence, outputting a verdict (`supported`, `unsupported`, or `ambiguous`) alongside an explanation.
- `stages/query_rewriter.py`: Triggered when verification fails. It inspects the critique, extracts missing clinical concepts, and generates an updated search query for another retrieval pass.

### 2.3 Memory Management (`memory/`)

The `memory/` package implements context persistence across interactive chat turns:

- `memory/short_term.py`: A rolling session buffer (`SessionBuffer`) that stores the last N dialogue turns to maintain natural conversational continuity within an active session.
- `memory/long_term.py`: A vector-backed persistent store (`LongTermMemory`). It analyzes incoming user statements, extracts critical patient facts (diagnoses, allergies, medications), saves them into ChromaDB, and retrieves relevant records in subsequent conversations.

---

## 3. Operational Directories & Resources

Supporting directories manage runtime configurations, vector databases, and experiment outputs.

### 3.1 Configuration Management (`configs/`)

YAML files in `configs/` define pipeline parameters for both benchmarks and interactive sessions:

- `configs/v0.yaml`: Baseline direct reasoning configuration.
- `configs/v1.yaml`: Retrieval-augmented pipeline with ChromaDB settings.
- `configs/v2.yaml` & `configs/v2-qr.yaml`: Self-correction pipelines with verification thresholds and query rewriting enabled.
- `configs/v3.yaml` & `configs/v3-qr.yaml`: Benchmark configurations with active long-term memory retrieval.
- `configs/v3-chat.yaml`: Tailored configuration for interactive multi-turn chat sessions.

### 3.2 Data Stores & Vector Indices (`data/`)

The `data/` directory stores persistent local database files:

- `data/chroma/`: Local ChromaDB vector database containing embedded medical textbook collections and user profile records.
- `data/medqa/`: Target evaluation datasets (such as MedQA-USMLE test splits) used for automated benchmarking.

### 3.3 Artifacts & Output Tracking (`output/`, `evaluate/`)

Artifacts generated during system execution are recorded and evaluated through dedicated directories:

- `output/`: Stores incremental run results as `.jsonl` files containing full episode traces, intermediate stage decisions, latency, and token consumption.
- `evaluate/evaluate.py`: The evaluation utility script. It parses prediction logs to compute accuracy, generates Markdown summary reports, and performs statistical hypothesis testing (McNemar's test and Bootstrap Confidence Intervals) to compare versions.

---

## 4. Execution Modes & Security Benchmark

The repository supports user-facing runtime modes as well as an adversarial security benchmarking subsystem.

### 4.1 Entrypoints & Operational Modes (`modes/`, `main.py`)

The root `main.py` serves as the primary CLI entrypoint, dispatching tasks to dedicated mode handlers:

- `main.py`: Command-line parser accepting `--mode` and `--config` flags.
- `modes/benchmark.py`: Runs batch evaluation over a dataset. It iterates through questions, passes them through the configured pipeline runner, and writes results to the output directory.
- `modes/chat.py`: An interactive CLI REPL allowing clinicians or users to converse with the system. It handles user input, memory retrieval, clinical reasoning, and memory updates in real time.

### 4.2 Attack Benchmark Integration (`attack/`)

The `attack/` directory houses a standalone prompt injection benchmark for evaluating system robustness against untrusted external inputs:

- `attack/cli.py`: Command-line runner for launching injection tests against any MED-AI configuration.
- `attack/attacks.py`: Implementation of five attack strategies (Naive, Escape Characters, Context Ignoring, Fake Completion, Combined).
- `attack/tasks.py`: Dataset loaders for remote classification tasks used as injection payloads (SST-2, SMS Spam, MRPC, HSOL, RTE).
- `attack/sampling.py`: Deterministic Cartesian product pairing and seed-based shuffling.
- `attack/benchmark.py`: Test runner that executes PNA-T, PNA-I, and attacked episodes while streaming incremental logs.
- `attack/configs/`: Specialized configuration overlays (`v0-attack.yaml`, `v1-qr-attack.yaml`, `v2-qr-attack.yaml`, `v3-qr-attack.yaml`).
