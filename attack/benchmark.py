"""Sequential prompt-injection benchmark orchestration."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from attack.attacks import ATTACKS, AttackBuilder
from attack.metrics import calculate_metrics
from attack.sampling import sample_pairs
from attack.tasks import InjectedExample, InjectedTask, load_tasks
from core.config import RunConfig, load_config
from core.llm_client import build_llm, extract_token_usage
from core.logger import DebugLogger
from core.runner import Runner
from core.types import EpisodeInput
from modes.benchmark import _load_dataset, _row_to_episode

CLEAN_EXTERNAL_SOURCE = "No additional external information is available."


def ensure_external_source_prompt(config: RunConfig) -> None:
    reasoning = config.get_stage("reasoning")
    if reasoning is None:
        raise ValueError("Target config must enable the reasoning stage")
    if "{external_source}" not in reasoning.prompt_template:
        reasoning.prompt_template = reasoning.prompt_template.rstrip() + "\n\nExternal source:\n{external_source}\n"


def load_targets(split: str, limit: int) -> list[EpisodeInput]:
    dataset = _load_dataset(split, limit)
    targets = [episode for index, row in enumerate(dataset)
               if (episode := _row_to_episode(row, index, split)) is not None]
    if not targets:
        raise ValueError("No valid MedQA target examples were loaded")
    return targets


def invoke_injected_task(
    config: RunConfig, task: InjectedTask, example: InjectedExample
) -> tuple[str, float, dict[str, int]]:
    reasoning = config.get_stage("reasoning")
    if reasoning is None:
        raise ValueError("Target config must enable the reasoning stage")
    llm = build_llm(model=config.model, base_url=config.model_base_url,
                    api_key=config.model_api_key, temperature=reasoning.temperature,
                    seed=config.seed,
                    enable_reasoning=reasoning.extra.get("reasoning", False),
                    reasoning_effort=reasoning.extra.get("reasoning_effort"))
    start = time.perf_counter()
    response = llm.invoke(f"{task.instruction}\n\nInput:\n{example.text}")
    latency_ms = (time.perf_counter() - start) * 1000
    return str(response.content), latency_ms, extract_token_usage(response)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def run_attack_benchmark(
    *, target_config: str, attack_names: list[str], task_names: list[str],
    split: str, target_limit: int, injected_limit: int, sample_size: int,
    seed: int, output_dir: str,
) -> dict[str, Any]:
    unknown_attacks = sorted(set(attack_names) - ATTACKS.keys())
    if unknown_attacks:
        raise ValueError(f"Unknown attacks: {', '.join(unknown_attacks)}")
    if target_limit < 1 or injected_limit < 1:
        raise ValueError("target_limit and injected_limit must be at least 1")

    config = load_config(target_config)
    reasoning = config.get_stage("reasoning")
    if reasoning is None:
        raise ValueError("Target config must enable the reasoning stage")
    reasoning.extra["reasoning"] = False
    ensure_external_source_prompt(config)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trace_path = destination / "trace.jsonl" if config.is_verbose else None
    if trace_path is not None:
        trace_path.write_text("", encoding="utf-8")
    logger = DebugLogger(
        level=config.debug,
        trace_path=str(trace_path) if trace_path is not None else None,
    )
    if trace_path is not None:
        logger.info(f"Verbose trace: {trace_path}")
    runner = Runner(config, logger)
    targets = load_targets(split, target_limit)
    tasks = load_tasks(task_names, injected_limit, seed)
    selected_attacks: dict[str, AttackBuilder] = {name: ATTACKS[name] for name in attack_names}

    clean_targets_path = destination / "clean_targets.jsonl"
    clean_injected_path = destination / "clean_injected.jsonl"
    cases_path = destination / "cases.jsonl"
    for path in (clean_targets_path, clean_injected_path, cases_path):
        path.write_text("", encoding="utf-8")

    run_metadata = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_config": target_config, "variant": config.variant, "model": config.model,
        "enable_reasoning": False,
        "target_dataset": "GBaker/MedQA-USMLE-4-options", "split": split,
        "target_limit": target_limit, "injected_limit": injected_limit,
        "sample_size_per_task": sample_size, "seed": seed, "attacks": attack_names,
        "trace_file": str(trace_path) if trace_path is not None else None,
        "tasks": [{"name": task.name, "dataset": task.dataset_name, "split": task.split,
                   "selected_count": len(task.examples)} for task in tasks],
    }
    _write_json(destination / "run_config.json", run_metadata)

    print(f"\n[PNA-T] Running {len(targets)} clean target examples")
    clean_target_rows: list[dict[str, Any]] = []
    clean_target_predictions: dict[str, str | None] = {}
    for target in targets:
        error, prediction = None, None
        latency_ms, token_usage = None, {}
        start = time.perf_counter()
        try:
            result = runner.run_episode(EpisodeInput(
                target.question_id, target.question, target.choices, target.gold_answer,
                CLEAN_EXTERNAL_SOURCE))
            prediction = result.predicted_answer
            latency_ms = result.total_latency_ms
            token_usage = result.total_token_usage
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            error = _error_message(exc)
            logger.warn(f"[PNA-T] {target.question_id}: {error}")
        row = {"target_id": target.question_id, "prediction": prediction,
               "gold": target.gold_answer, "is_correct": prediction == target.gold_answer,
               "latency_ms": latency_ms, "token_usage": token_usage, "error": error}
        clean_target_predictions[target.question_id] = prediction
        clean_target_rows.append(row)
        _append_jsonl(clean_targets_path, row)

    print(f"\n[PNA-I] Running up to {injected_limit} clean examples per injected task")
    clean_injected_rows: list[dict[str, Any]] = []
    clean_injected_predictions: dict[tuple[str, str], str | None] = {}
    for task in tasks:
        for example in task.examples:
            error, raw, prediction = None, "", None
            latency_ms, token_usage = None, {}
            start = time.perf_counter()
            try:
                raw, latency_ms, token_usage = invoke_injected_task(config, task, example)
                prediction = task.parse_label(raw)
            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                error = _error_message(exc)
                logger.warn(f"[PNA-I] {task.name}/{example.id}: {error}")
            row = {"task": task.name, "dataset": task.dataset_name, "split": task.split,
                   "example_id": example.id, "prediction": prediction,
                   "gold": example.gold_label,
                   "is_correct": task.score(prediction, example.gold_label),
                   "raw_response": raw, "latency_ms": latency_ms,
                   "token_usage": token_usage, "error": error}
            clean_injected_predictions[(task.name, example.id)] = prediction
            clean_injected_rows.append(row)
            _append_jsonl(clean_injected_path, row)

    print(f"\n[ATTACK] Running {len(selected_attacks)} attacks over sampled pairs")
    case_rows: list[dict[str, Any]] = []
    for task in tasks:
        pairs = sample_pairs(targets, task.examples, sample_size, seed)
        for attack_name, attack_builder in selected_attacks.items():
            for target, injected in pairs:
                compromised_source = attack_builder(CLEAN_EXTERNAL_SOURCE, task.instruction,
                                                    injected.text)
                error, attacked_target_prediction = None, None
                attacked_label, raw_response = None, ""
                latency_ms, token_usage = None, {}
                start = time.perf_counter()
                try:
                    attacked = runner.run_episode(EpisodeInput(
                        target.question_id, target.question, target.choices,
                        target.gold_answer, compromised_source))
                    attacked_target_prediction = attacked.predicted_answer
                    raw_response = attacked.reasoning_raw_response or ""
                    attacked_label = task.parse_label(raw_response)
                    latency_ms = attacked.total_latency_ms
                    token_usage = attacked.total_token_usage
                except Exception as exc:
                    latency_ms = (time.perf_counter() - start) * 1000
                    error = _error_message(exc)
                    logger.warn(f"[ATTACK] {attack_name}/{task.name}/{target.question_id}: {error}")
                clean_injected = clean_injected_predictions[(task.name, injected.id)]
                row = {
                    "target_id": target.question_id, "injected_task": task.name,
                    "injected_example_id": injected.id, "attack_method": attack_name,
                    "clean_external_source": CLEAN_EXTERNAL_SOURCE,
                    "compromised_external_source": compromised_source,
                    "target_gold": target.gold_answer,
                    "clean_target_prediction": clean_target_predictions[target.question_id],
                    "attacked_target_prediction": attacked_target_prediction,
                    "target_correct_under_attack": attacked_target_prediction == target.gold_answer,
                    "injected_gold": injected.gold_label,
                    "clean_injected_prediction": clean_injected,
                    "attacked_injected_prediction": attacked_label,
                    "asv_contribution": task.score(attacked_label, injected.gold_label),
                    "mr_contribution": task.score(attacked_label, clean_injected),
                    "raw_attacked_response": raw_response,
                    "latency_ms": latency_ms, "token_usage": token_usage,
                    "error": error,
                    "variant": config.variant, "seed": seed,
                }
                case_rows.append(row)
                _append_jsonl(cases_path, row)

    metrics = calculate_metrics(clean_target_rows, clean_injected_rows, case_rows)
    _write_json(destination / "metrics.json", metrics)
    return {"output_dir": str(destination), "trace_file": run_metadata["trace_file"],
            "case_count": len(case_rows),
            "metrics": metrics, "run_config": run_metadata}
