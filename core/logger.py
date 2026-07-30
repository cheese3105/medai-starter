"""Logger — terminal output + JSONL writer + verbose trace writer."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from core.types import EpisodeResult


def make_output_path(output_dir: str, variant: str, kind: str = "predictions",
                     timestamp: Optional[int] = None) -> str:
    ts = timestamp if timestamp is not None else int(time.time())
    return str(Path(output_dir) / f"{kind}_{variant}_{ts}.jsonl")


class DebugLogger:
    """In tiến trình lên terminal. Level: "off" | "info" | "verbose"."""

    def __init__(self, level: str = "off", trace_path: Optional[str] = None):
        self.level = level
        self.enabled = level in ("info", "verbose")
        self.verbose = level == "verbose"
        self._trace_writer: Optional[TraceWriter] = None
        if self.verbose and trace_path:
            self._trace_writer = TraceWriter(trace_path)

    def stage_start(self, question_id: str, iteration: int, stage_name: str,
                    extra: str = "") -> None:
        if not self.enabled:
            return
        iter_str = f"iter={iteration} → " if iteration > 0 else ""
        extra_str = f"  ({extra})" if extra else ""
        print(f"  [{question_id}] {iter_str}{stage_name}{extra_str}", flush=True)

    def stage_end(self, question_id: str, iteration: int, stage_name: str,
                  summary: str, latency_ms: float) -> None:
        if not self.enabled:
            return
        iter_str = f"iter={iteration} → " if iteration > 0 else ""
        print(f"  [{question_id}] {iter_str}{stage_name} → {summary}  {latency_ms:.0f}ms", flush=True)

    def episode_done(self, question_id: str, answer: str,
                     iterations: int, cost: Optional[float]) -> None:
        if not self.enabled:
            return
        cost_str = f", cost=${cost:.4f}" if cost is not None else ""
        print(f"  [{question_id}] DONE → answer=\"{answer}\" (iterations={iterations}{cost_str})", flush=True)

    def memory_info(self, question_id: str, msg: str) -> None:
        if not self.enabled:
            return
        print(f"  [{question_id}] memory: {msg}", flush=True)

    def warn(self, msg: str) -> None:
        """Luôn in ra bất kể level."""
        print(f"  [WARN] {msg}", file=sys.stderr, flush=True)

    def info(self, msg: str) -> None:
        if not self.enabled:
            return
        print(f"  [INFO] {msg}", flush=True)

    def trace(self, question_id: str, iteration: int, stage_name: str,
              question: str, choices: str, prompt: str, raw_response: str,
              parsed_data: dict, latency_ms: float, token_usage: dict) -> None:
        """Ghi full prompt/response — chỉ khi level="verbose" và trace_path set."""
        if self._trace_writer is None:
            return
        self._trace_writer.write({
            "question_id": question_id, "iteration": iteration,
            "stage_name": stage_name, "question": question, "choices": choices,
            "prompt": prompt, "raw_response": raw_response,
            "parsed_data": parsed_data, "latency_ms": latency_ms,
            "token_usage": token_usage,
        })


class TraceWriter:
    def __init__(self, output_path: str):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def write(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class PredictionWriter:
    """Ghi EpisodeResult ra JSONL, kèm header comment mô tả config."""

    def __init__(self, output_path: str):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def write_header(self, summary: dict[str, Any]) -> None:
        """Header dạng comment ở đầu file, để biết file chạy bằng config gì."""
        lines = [
            "# " + "=" * 60,
            f"# MED-AI predictions — variant={summary.get('variant')}",
            f"# generated_at: {summary.get('generated_at')}",
            f"# models: {summary.get('models')}",
            f"# split: {summary.get('split')}  limit: {summary.get('limit')}",
            f"# pipeline: {summary.get('pipeline_summary')}",
            f"# memory: {summary.get('memory_summary')}",
            "# " + "=" * 60,
        ]
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def write(self, result: EpisodeResult) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")