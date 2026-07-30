"""Benchmark mode — chạy MedQA-USMLE qua pipeline, ghi predictions JSONL."""

from __future__ import annotations

import time

from core.config import RunConfig
from core.logger import DebugLogger, PredictionWriter, make_output_path
from core.runner import Runner
from core.types import EpisodeInput

LABELS = ["A", "B", "C", "D"]
OUTPUT_DIR = "output"


def _load_dataset(split: str, limit: int | None = None):
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    if limit is not None and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def _row_to_episode(row: dict, index: int, split: str) -> EpisodeInput | None:
    question = row.get("question")
    options = row.get("options")
    answer_idx = row.get("answer_idx")
    if not question or not options:
        return None
    for label in LABELS:
        if label not in options:
            return None
    choices = [options[label] for label in LABELS]
    return EpisodeInput(
        question_id=f"{split}_{index:05d}",
        question=question, choices=choices, gold_answer=answer_idx,
    )


def _model_summary(config: RunConfig) -> str:
    """Liệt kê agent → model, chỉ hiển thị agent đang enabled."""
    entries = []
    for s in config.pipeline:
        if not s.enabled:
            continue
        if s.name == "reasoning":
            entries.append(f"reasoning: {config.model}")
        elif s.name == "verifier":
            entries.append(f"verifier: {config.verifier_model}")
        elif s.name == "query_rewriter":
            entries.append(f"query_rewriter: {config.query_rewriter_model}")
        elif s.name == "retrieval":
            entries.append(f"retrieval: {config.embedding_model or '(embedding)'}")
    return ", ".join(entries)


def _pipeline_summary(config: RunConfig) -> str:
    parts = []
    for s in config.pipeline:
        if not s.enabled:
            continue
        if s.name == "retrieval":
            parts.append(f"retrieval(top_k={s.top_k})")
        elif s.name == "verifier":
            parts.append(f"verifier(max_iterations={s.max_iterations})")
        else:
            parts.append(s.name)
    return " -> ".join(parts) if parts else "(rỗng)"


def _memory_summary(config: RunConfig) -> str:
    parts = []
    if config.memory.short_term.enabled:
        parts.append(f"STM(scope={config.memory.short_term.scope})")
    if config.memory.long_term.enabled:
        parts.append(f"LTM(mode={config.memory.long_term.mode})")
    return " + ".join(parts) if parts else "OFF"


def run_benchmark(
    config: RunConfig,
    split: str = "test",
    limit: int | None = None,
    output_path: str | None = None,
) -> dict:
    logger = DebugLogger(level=config.debug)

    if config.memory.long_term.enabled:
        config.memory.long_term.mode = "read_only"
        config.memory.long_term.write_after_answer = False
        logger.info("LTM mode ép về read_only cho benchmark")

    timestamp = int(time.time())
    if output_path is None:
        output_path = make_output_path(OUTPUT_DIR, config.variant, "predictions", timestamp)

    trace_path = None
    if config.is_verbose:
        trace_path = make_output_path(OUTPUT_DIR, config.variant, "trace", timestamp)
        logger = DebugLogger(level=config.debug, trace_path=trace_path)

    model_info = _model_summary(config)

    runner = Runner(config, logger)
    writer = PredictionWriter(output_path)
    writer.write_header({
        "variant": config.variant,
        "generated_at": timestamp,
        "models": model_info,
        "split": split,
        "limit": limit,
        "pipeline_summary": _pipeline_summary(config),
        "memory_summary": _memory_summary(config),
    })

    logger.info(f"Loading MedQA split={split}" + (f" limit={limit}" if limit else ""))
    dataset = _load_dataset(split, limit)
    logger.info(f"Loaded {len(dataset)} questions")
    if trace_path:
        logger.info(f"Verbose trace: {trace_path}")

    total, correct, skipped = 0, 0, 0

    for i, row in enumerate(dataset):
        episode = _row_to_episode(row, i, split)
        if episode is None:
            skipped += 1
            logger.warn(f"Dòng {i} bị skip do thiếu question/options")
            continue
        try:
            result = runner.run_episode(episode)
            writer.write(result)
            total += 1
            if result.is_correct:
                correct += 1
            if not config.is_debug:
                print(
                    f"\r  [{total}/{len(dataset)}] acc={correct/total:.1%} "
                    f"(last: {result.predicted_answer}={'✓' if result.is_correct else '✗'})",
                    end="", flush=True,
                )
        except Exception as e:
            logger.warn(f"Lỗi câu {episode.question_id}: {e}")
            skipped += 1

    if not config.is_debug:
        print()

    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"  Variant:  {config.variant}")
    print(f"  Models:")
    for s in config.pipeline:
        if not s.enabled:
            continue
        if s.name == "reasoning":
            print(f"    reasoning:      {config.model}")
        elif s.name == "verifier":
            print(f"    verifier:       {config.verifier_model}")
        elif s.name == "query_rewriter":
            print(f"    query_rewriter: {config.query_rewriter_model}")
        elif s.name == "retrieval":
            print(f"    retrieval:      {config.embedding_model or '(embedding)'}")
    print(f"  Split:    {split}")
    print(f"  Total:    {total}")
    print(f"  Correct:  {correct}")
    print(f"  Accuracy: {accuracy:.1%}")
    if skipped:
        print(f"  Skipped:  {skipped}")
    print(f"  Output:   {output_path}")
    if trace_path:
        print(f"  Trace:    {trace_path}")
    print(f"{'='*50}")

    return {
        "variant": config.variant, "models": model_info,
        "split": split, "total": total, "correct": correct,
        "accuracy": accuracy, "skipped": skipped,
        "output_file": output_path, "trace_file": trace_path,
    }