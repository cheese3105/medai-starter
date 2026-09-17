"""Runner — controller loop bao ngoài pipeline.

Không có verifier → chạy 1 vòng (V0/V1).
Có verifier → lặp đến verdict="supported" hoặc max_iterations (V2/V3).

Query-First Pipeline (v2-qr / v3-qr):
  Mỗi iteration: QueryRewriter → Retrieval → Reasoning → Verifier
  - QueryRewriter chạy ĐẦU TIÊN để sinh retrieval_query từ question gốc.
  - question gốc KHÔNG BAO GIỜ bị ghi đè; chỉ retrieval_query thay đổi.
  - query_rewrite_count chỉ đếm retry (iter 2+), không tính iter 1.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from core.config import RunConfig
from core.logger import DebugLogger
from core.types import EpisodeInput, EpisodeResult, IterationRecord

_STAGE_CLASSES: dict[str, type] = {}
LABELS = ["A", "B", "C", "D"]


def _get_stage_class(name: str):
    if not _STAGE_CLASSES:
        from stages.reasoning import ReasoningStage
        from stages.retrieval import RetrievalStage
        from stages.verifier import VerifierStage
        from stages.query_rewriter import QueryRewriterStage
        _STAGE_CLASSES.update({
            "reasoning": ReasoningStage, "retrieval": RetrievalStage,
            "verifier": VerifierStage, "query_rewriter": QueryRewriterStage,
        })
    return _STAGE_CLASSES.get(name)


def _format_choices(choices: list[str]) -> str:
    if not choices:
        return ""
    return "\n".join(f"{label}. {text}" for label, text in zip(LABELS, choices))


def _estimate_cost(token_usage: dict, pricing_in: Optional[float], pricing_out: Optional[float]) -> Optional[float]:
    if pricing_in is None or pricing_out is None:
        return None
    return token_usage.get("input", 0) / 1000 * pricing_in + token_usage.get("output", 0) / 1000 * pricing_out


class Runner:
    def __init__(self, run_config: RunConfig, logger: DebugLogger):
        self.config = run_config
        self.logger = logger
        self.stages: dict[str, Any] = {}
        for sc in run_config.pipeline:
            if not sc.enabled:
                continue
            cls = _get_stage_class(sc.name)
            if cls is None:
                raise ValueError(f"Stage '{sc.name}' chưa được implement.")
            self.stages[sc.name] = cls(run_config, sc)
        self.stage_order = [sc.name for sc in run_config.pipeline if sc.enabled]
        self.has_verifier = "verifier" in self.stages
        self.has_retrieval = "retrieval" in self.stages
        self.has_query_rewriter = "query_rewriter" in self.stages
        self.max_iterations = run_config.max_iterations
        self.stm_loop_enabled = (
            run_config.memory.short_term.enabled
            and run_config.memory.short_term.scope in ("loop", "both")
        )

    def run_episode(self, episode_input: EpisodeInput, *,
                    stm_session_history: Optional[str] = None,
                    ltm_facts: Optional[str] = None) -> EpisodeResult:
        qid = episode_input.question_id
        ep_start = time.time()

        base_context: dict[str, Any] = {
            "question":          episode_input.question,           # KHÔNG BAO GIỜ thay đổi
            "retrieval_query":   episode_input.question,           # cập nhật bởi QR mỗi iter
            "choices":           _format_choices(episode_input.choices),
            "evidence":          "",   # cập nhật sau mỗi iter fail (để QR biết evidence cũ)
            "draft_explanation": "",   # lý do reject từ verifier, cho QR ở iter tiếp
            "loop_history":      "",
            "ltm_facts":         ltm_facts or "",
            "stm_history":       stm_session_history or "",
        }

        total_tokens = {"input": 0, "output": 0}
        ret_latency, ret_tokens = 0.0, {"input": 0, "output": 0}
        rsn_latency, rsn_tokens = 0.0, {"input": 0, "output": 0}
        ver_latency, ver_tokens = 0.0, {"input": 0, "output": 0}
        rw_latency, rw_tokens = 0.0, {"input": 0, "output": 0}
        rw_count = 0          # chỉ đếm retry rewrites (iter 2+, sau failed verification)
        initial_qr_ran = False  # True nếu QR chạy ở iter 1
        evidence_ids: list[str] = []
        verdict_history: list[str] = []
        loop_scratchpad: list[str] = []
        final_answer, final_explanation, final_confidence = "INVALID", "", 0.0
        reasoning_raw_response = ""
        stopped_early = False

        for iteration in range(1, self.max_iterations + 1):
            context = dict(base_context)
            context["loop_history"] = "\n".join(loop_scratchpad) if loop_scratchpad else ""

            # BƯỚC 1: Query Rewriter (đầu mỗi iteration, trước retrieval)
            if self.has_query_rewriter:
                self.logger.stage_start(qid, iteration, "query_rewriter")
                rw_out = self.stages["query_rewriter"].run(context)
                retrieval_query = rw_out.data.get("rewritten_query", context["question"])
                rw_latency += rw_out.latency_ms
                for k in ("input", "output"):
                    rw_tokens[k] += rw_out.token_usage.get(k, 0)
                    total_tokens[k] += rw_out.token_usage.get(k, 0)
                self.logger.stage_end(qid, iteration, "query_rewriter",
                                      f'query="{retrieval_query[:60]}..."' if len(retrieval_query) > 60
                                      else f'query="{retrieval_query}"',
                                      rw_out.latency_ms)
                if self.logger.verbose:
                    self.logger.trace(qid, iteration, "query_rewriter",
                                      context["question"], context["choices"],
                                      rw_out.data.get("prompt", ""), rw_out.data.get("raw_response", ""),
                                      rw_out.data, rw_out.latency_ms, rw_out.token_usage)
                context["retrieval_query"] = retrieval_query
                # iter 1 = initial formulation (không tính retry), iter 2+ = retry
                if iteration == 1:
                    initial_qr_ran = True
                else:
                    rw_count += 1

            # BƯỚC 2: Retrieval (dùng retrieval_query, không phải question gốc)
            if self.has_retrieval:
                self.logger.stage_start(qid, iteration, "retrieval", f"top_k={self.stages['retrieval'].top_k}")
                out = self.stages["retrieval"].run(context)
                context["evidence"] = out.data.get("evidence", "")
                evidence_ids = out.data.get("evidence_ids", [])
                ret_latency += out.latency_ms
                self.logger.stage_end(qid, iteration, "retrieval", f"{out.data.get('num_results', 0)} docs", out.latency_ms)
                if self.logger.verbose:
                    self.logger.trace(qid, iteration, "retrieval",
                                      context["question"], context["choices"],
                                      out.data.get("prompt", ""), out.data.get("raw_response", ""),
                                      out.data, out.latency_ms, out.token_usage)

            # BƯỚC 3: Reasoning (luôn nhận question gốc). external_source chỉ
            # được thêm vào context của reasoning, không lộ sang stage khác.
            reasoning_context = dict(context)
            reasoning_context["external_source"] = episode_input.external_source
            self.logger.stage_start(qid, iteration, "reasoning")
            out = self.stages["reasoning"].run(reasoning_context)
            draft_answer = out.data.get("answer", "INVALID")
            draft_explanation = out.data.get("explanation", "")
            draft_confidence = out.data.get("confidence", 0.0)
            reasoning_raw_response = out.data.get("raw_response", "")
            rsn_latency += out.latency_ms
            for k in ("input", "output"):
                rsn_tokens[k] += out.token_usage.get(k, 0)
                total_tokens[k] += out.token_usage.get(k, 0)
            self.logger.stage_end(qid, iteration, "reasoning", f'draft="{draft_answer}" (conf={draft_confidence:.2f})', out.latency_ms)
            if self.logger.verbose:
                self.logger.trace(qid, iteration, "reasoning",
                                  reasoning_context["question"], reasoning_context["choices"],
                                  out.data.get("prompt", ""), out.data.get("raw_response", ""),
                                  out.data, out.latency_ms, out.token_usage)

            # BƯỚC 4: Verifier (luôn nhận question gốc)
            if self.has_verifier:
                context.update(draft_answer=draft_answer, draft_explanation=draft_explanation,
                               draft_confidence=draft_confidence)
                self.logger.stage_start(qid, iteration, "verifier")
                out = self.stages["verifier"].run(context)
                verdict = out.data.get("verdict", "supported")
                verdict_history.append(verdict)
                ver_latency += out.latency_ms
                for k in ("input", "output"):
                    ver_tokens[k] += out.token_usage.get(k, 0)
                    total_tokens[k] += out.token_usage.get(k, 0)
                self.logger.stage_end(qid, iteration, "verifier", f"verdict={verdict}", out.latency_ms)
                if self.logger.verbose:
                    self.logger.trace(qid, iteration, "verifier",
                                      context["question"], context["choices"],
                                      out.data.get("prompt", ""), out.data.get("raw_response", ""),
                                      out.data, out.latency_ms, out.token_usage)
                final_answer = out.data.get("final_answer", draft_answer)
                final_explanation = out.data.get("explanation", draft_explanation)
                final_confidence = out.data.get("confidence", draft_confidence)

                if verdict == "supported":
                    break

                # Lưu evidence vừa fail + lý do reject để QR dùng ở iter tiếp theo
                base_context["evidence"] = context["evidence"]
                base_context["draft_explanation"] = out.data.get("explanation", draft_explanation)
                # Fallback (không có QR): dùng suggested_query từ verifier nếu có
                if not self.has_query_rewriter:
                    suggested = out.data.get("suggested_query", "")
                    base_context["retrieval_query"] = suggested if suggested else context["question"]

                if self.stm_loop_enabled:
                    entry = f"[iter {iteration}] answer={draft_answer}, verdict=unsupported, reason={out.data.get('explanation', '')}"
                    loop_scratchpad.append(entry)
            else:
                final_answer, final_explanation, final_confidence = draft_answer, draft_explanation, draft_confidence
                break
        else:
            stopped_early = True

        total_latency = (time.time() - ep_start) * 1000
        is_correct = (final_answer == episode_input.gold_answer) if episode_input.gold_answer is not None else None
        self.logger.episode_done(qid, final_answer, len(verdict_history) or 1,
                                 _estimate_cost(total_tokens, self.config.pricing.input_per_1k, self.config.pricing.output_per_1k))

        return EpisodeResult(
            question_id=qid, variant=self.config.variant, model=self.config.model,
            predicted_answer=final_answer, explanation=final_explanation,
            confidence=final_confidence, gold_answer=episode_input.gold_answer,
            is_correct=is_correct, total_latency_ms=total_latency,
            total_token_usage=total_tokens,
            estimated_cost=_estimate_cost(total_tokens, self.config.pricing.input_per_1k, self.config.pricing.output_per_1k),
            evidence_used=evidence_ids if self.has_retrieval else None,
            retrieval_latency_ms=ret_latency if self.has_retrieval else None,
            retrieval_token_usage=ret_tokens if self.has_retrieval else None,
            iteration_count=len(verdict_history) if self.has_verifier else None,
            verifier_verdict=verdict_history[-1] if verdict_history else None,
            verdict_history=verdict_history if verdict_history else None,
            stopped_after_max_iterations=stopped_early if self.has_verifier else None,
            reasoning_latency_ms=rsn_latency, reasoning_token_usage=rsn_tokens,
            reasoning_raw_response=reasoning_raw_response,
            verifier_latency_ms=ver_latency if self.has_verifier else None,
            verifier_token_usage=ver_tokens if self.has_verifier else None,
            query_rewrite_count=rw_count if self.has_query_rewriter else None,
            rewriter_latency_ms=rw_latency if self.has_query_rewriter else None,
            rewriter_token_usage=rw_tokens if self.has_query_rewriter else None,
            initial_query_generated=initial_qr_ran if self.has_query_rewriter else None,
            loop_history_length=len(loop_scratchpad) if loop_scratchpad else None,
        )
