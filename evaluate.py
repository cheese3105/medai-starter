#!/usr/bin/env python3
"""
evaluate.py — Script đánh giá hệ thống Multi-Agent LLM cho MedQA-USMLE
========================================================================

Đọc các file prediction (.jsonl) do từng biến thể hệ thống (V0, V1, V2, V3, V3-qr, V4...)
sinh ra, tính toán đầy đủ các chỉ số theo yêu cầu đề bài (mục 5 - Metrics):

    - Accuracy = số câu đúng / tổng số câu
    - Invalid response rate = tỷ lệ câu trả lời không hợp lệ
    - Accuracy gain = Accuracy(variant) - Accuracy(baseline)
    - Win / Loss / Tie giữa 2 biến thể (so từng câu)
    - McNemar's test (kiểm định có ý nghĩa thống kê giữa 2 biến thể)
    - Bootstrap 95% CI cho accuracy và cho accuracy gain
    - Cost & latency (tổng token, ước tính chi phí, độ trễ trung bình/median)

USAGE
-----
    # Chỉ định file cụ thể
    python evaluate.py --files predictions_v0.jsonl predictions_v3-qr.jsonl

    # Hoặc để script tự quét thư mục (mặc định ./predictions hoặc thư mục hiện tại)
    python evaluate.py --dir ./predictions

    # Chỉ định biến thể nào là baseline (mặc định: variant có tên chứa "v0")
    python evaluate.py --dir . --baseline v0

    # Xuất thêm báo cáo Markdown
    python evaluate.py --dir . --report report.md

Output:
    - In bảng tổng hợp ra console
    - Lưu summary_metrics.csv (accuracy, invalid rate, cost, latency... theo từng variant)
    - Lưu pairwise_comparison.csv (win/loss/tie, McNemar, bootstrap CI giữa các cặp)
    - (tuỳ chọn) report.md — báo cáo Markdown format đẹp để đưa thẳng vào report
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.stats import binomtest

# ---------------------------------------------------------------------------
# Cấu hình chi phí ước tính (USD / 1M token) — CHỈNH LẠI cho đúng model bạn dùng.
# Để trống / None nếu không muốn ước tính cost (script sẽ bỏ qua và ghi "N/A").
# ---------------------------------------------------------------------------
MODEL_PRICING_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    # "deepseek/deepseek-v4-flash": {"input": 0.27, "output": 1.10},
    # Thêm model của bạn ở đây, ví dụ:
    # "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

VALID_ANSWERS = {"A", "B", "C", "D", "E"}  # MedQA-USMLE chuẩn là 4 đáp án A-D,
                                             # để E dự phòng nếu dataset mở rộng.

N_BOOTSTRAP = 10_000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Record:
    """Một dòng prediction đã parse."""
    question_id: str
    variant: str
    predicted_answer: Optional[str]
    gold_answer: Optional[str]
    is_correct: Optional[bool]
    explanation: str = ""
    confidence: Optional[float] = None
    total_latency_ms: Optional[float] = None
    total_token_usage: dict = field(default_factory=dict)
    estimated_cost: Optional[float] = None
    model: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Câu trả lời hợp lệ = predicted_answer nằm trong tập đáp án cho phép."""
        return self.predicted_answer in VALID_ANSWERS

    @property
    def correct(self) -> bool:
        """
        Chuẩn hoá độ đúng/sai. Ưu tiên field `is_correct` có sẵn trong file
        (do hệ thống tự chấm khi sinh prediction). Nếu thiếu, tự so sánh
        predicted_answer với gold_answer. Nếu không có gold_answer -> False
        (không thể chấm) và câu này sẽ được tính là invalid/không chấm được.
        """
        if self.is_correct is not None:
            return bool(self.is_correct)
        if self.gold_answer is not None and self.predicted_answer is not None:
            return self.predicted_answer == self.gold_answer
        return False


@dataclass
class VariantData:
    """Toàn bộ record của 1 biến thể, đã lập chỉ mục theo question_id."""
    variant: str
    filepath: str
    header_meta: dict = field(default_factory=dict)  # thông tin đọc từ dòng comment '#'
    records: dict[str, Record] = field(default_factory=dict)  # question_id -> Record

    def __len__(self):
        return len(self.records)


# ---------------------------------------------------------------------------
# Đọc file
# ---------------------------------------------------------------------------

def parse_header_comments(lines: list[str]) -> dict:
    """
    Các file prediction có header dạng comment '#':
        # ============================================================
        # MED-AI predictions — variant=v3-qr
        # generated_at: 1785466972
        # models: retrieval: bge-m3, reasoning: deepseek/deepseek-v4-flash, ...
        # split: test  limit: 500
        # pipeline: retrieval(top_k=5) -> reasoning -> verifier(...) -> query_rewriter
        # memory: STM(scope=loop) + LTM(mode=read_only)
        # ============================================================
    Trích xuất các dòng này thành dict metadata để hiển thị trong report,
    không bắt buộc phải có — nếu thiếu, trả về dict rỗng.
    """
    meta: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line.startswith("#"):
            continue
        content = line.lstrip("#").strip()
        if not content or set(content) == {"="}:
            continue
        if ":" in content:
            key, _, val = content.partition(":")
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            if key in meta:  # nếu key trùng, nối thêm
                meta[key] += " | " + val
            else:
                meta[key] = val
        else:
            meta.setdefault("_notes", []).append(content) if isinstance(
                meta.get("_notes"), list
            ) else meta.setdefault("_notes", [content])
    return meta


def load_jsonl_variant(filepath: str, variant_override: Optional[str] = None) -> VariantData:
    """Đọc 1 file .jsonl, bỏ qua dòng comment '#', parse từng dòng JSON thành Record."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    header_meta = parse_header_comments([l for l in raw_lines if l.strip().startswith("#")])

    records: dict[str, Record] = {}
    variant_name = variant_override
    n_bad_lines = 0

    for lineno, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            n_bad_lines += 1
            continue

        qid = obj.get("question_id")
        if qid is None:
            n_bad_lines += 1
            continue

        v = obj.get("variant") or variant_name or os.path.basename(filepath)
        if variant_name is None:
            variant_name = v

        rec = Record(
            question_id=str(qid),
            variant=str(v),
            predicted_answer=obj.get("predicted_answer"),
            gold_answer=obj.get("gold_answer"),
            is_correct=obj.get("is_correct"),
            explanation=obj.get("explanation", "") or "",
            confidence=obj.get("confidence"),
            total_latency_ms=obj.get("total_latency_ms"),
            total_token_usage=obj.get("total_token_usage") or {},
            estimated_cost=obj.get("estimated_cost"),
            model=obj.get("model", ""),
            raw=obj,
        )

        if qid in records:
            # Trùng question_id trong cùng file -> giữ bản ghi cuối, cảnh báo
            print(f"  [CẢNH BÁO] {filepath}:{lineno} — question_id trùng lặp "
                  f"'{qid}', dùng bản ghi sau đè bản ghi trước.", file=sys.stderr)
        records[str(qid)] = rec

    if n_bad_lines:
        print(f"  [CẢNH BÁO] {filepath}: bỏ qua {n_bad_lines} dòng lỗi/không parse được.",
              file=sys.stderr)

    return VariantData(
        variant=variant_name or os.path.basename(filepath),
        filepath=filepath,
        header_meta=header_meta,
        records=records,
    )


def discover_files(explicit_files: list[str], directory: Optional[str]) -> list[str]:
    """Ưu tiên --files; nếu rỗng thì scan --dir (mặc định thư mục hiện tại) tìm *.jsonl."""
    files: list[str] = []
    if explicit_files:
        for pattern in explicit_files:
            matched = glob.glob(pattern)
            files.extend(matched if matched else [pattern])
    else:
        scan_dir = directory or "."
        files = sorted(glob.glob(os.path.join(scan_dir, "*.jsonl")))

    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(
            "Không tìm thấy file .jsonl nào. Dùng --files a.jsonl b.jsonl hoặc --dir <thư mục>."
        )
    return files


# ---------------------------------------------------------------------------
# Metrics: single-variant
# ---------------------------------------------------------------------------

def estimate_cost(record: Record) -> Optional[float]:
    """Ước tính cost nếu có bảng giá cho model; None nếu không có/không đủ dữ liệu."""
    if record.estimated_cost is not None:
        return record.estimated_cost
    pricing = MODEL_PRICING_PER_1M_TOKENS.get(record.model)
    usage = record.total_token_usage or {}
    if not pricing or "input" not in usage or "output" not in usage:
        return None
    return (usage["input"] / 1e6) * pricing["input"] + (usage["output"] / 1e6) * pricing["output"]


def limit_to_first_n_questions(variants: dict[str, "VariantData"], n: int) -> None:
    """
    Cắt MỌI biến thể trong `variants` xuống chỉ còn N câu hỏi đầu tiên, chọn theo
    question_id sắp xếp tăng dần trên HỢP các question_id xuất hiện ở bất kỳ biến thể nào
    (đảm bảo dùng đúng 1 bộ ID chung, không phải "N dòng đầu của mỗi file" một cách độc lập —
    vì thứ tự dòng có thể lệch nhau giữa các file).
    Sửa trực tiếp (in-place) từng VariantData.records.
    """
    all_ids: set[str] = set()
    for vdata in variants.values():
        all_ids |= set(vdata.records.keys())

    # Sắp xếp tự nhiên theo phần số trong question_id (vd test_00000, test_00001, ...);
    # fallback về sort chuỗi thường nếu id không theo dạng đó.
    def sort_key(qid: str):
        import re
        m = re.search(r"(\d+)", qid)
        return (int(m.group(1)) if m else float("inf"), qid)

    target_ids = set(sorted(all_ids, key=sort_key)[:n])

    for vdata in variants.values():
        before = len(vdata.records)
        vdata.records = {qid: rec for qid, rec in vdata.records.items() if qid in target_ids}
        after = len(vdata.records)
        print(f"  [--limit-first-n] '{vdata.variant}': {before} -> {after} câu "
              f"(giữ lại các câu nằm trong {n} ID đầu tiên).")



def compute_variant_summary(vdata: VariantData, official_n: Optional[int] = None) -> dict:
    """
    Tính các chỉ số tổng hợp cho 1 biến thể.
    official_n: nếu truyền vào (vd 1273 cho MedQA-USMLE full test set), dùng làm mẫu số
                chính thức cho accuracy thay vì len(records) — hữu ích khi 1 biến thể
                bị thiếu vài câu (lỗi API...) nhưng vẫn muốn accuracy tính trên full set.
    """
    recs = list(vdata.records.values())
    n_total = len(recs)
    denom = official_n if official_n else n_total

    n_valid = sum(1 for r in recs if r.is_valid)
    n_invalid = n_total - n_valid
    n_correct = sum(1 for r in recs if r.correct)

    accuracy = n_correct / denom if denom else float("nan")
    invalid_rate = n_invalid / n_total if n_total else float("nan")

    latencies = [r.total_latency_ms for r in recs if r.total_latency_ms is not None]
    input_tokens = [r.total_token_usage.get("input", 0) for r in recs if r.total_token_usage]
    output_tokens = [r.total_token_usage.get("output", 0) for r in recs if r.total_token_usage]
    costs = [c for c in (estimate_cost(r) for r in recs) if c is not None]

    return {
        "variant": vdata.variant,
        "filepath": vdata.filepath,
        "n_total": n_total,
        "n_denom_used_for_accuracy": denom,
        "n_correct": n_correct,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else None,
        "median_latency_ms": float(np.median(latencies)) if latencies else None,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "total_input_tokens": int(sum(input_tokens)) if input_tokens else None,
        "total_output_tokens": int(sum(output_tokens)) if output_tokens else None,
        "avg_tokens_per_question": (
            float(np.mean([i + o for i, o in zip(input_tokens, output_tokens)]))
            if input_tokens and output_tokens else None
        ),
        "total_estimated_cost_usd": float(sum(costs)) if costs else None,
        "avg_cost_per_question_usd": float(np.mean(costs)) if costs else None,
        "pipeline": vdata.header_meta.get("pipeline", ""),
        "memory": vdata.header_meta.get("memory", ""),
        "models": vdata.header_meta.get("models", ""),
    }


# ---------------------------------------------------------------------------
# Metrics: pairwise (so sánh 2 biến thể trên cùng tập câu hỏi)
# ---------------------------------------------------------------------------

def aligned_correctness(a: VariantData, b: VariantData) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Lấy giao (intersection) question_id giữa 2 biến thể, trả về:
        - danh sách question_id chung
        - mảng đúng/sai (1/0) của a theo đúng thứ tự đó
        - mảng đúng/sai (1/0) của b theo đúng thứ tự đó
    Cảnh báo nếu 2 tập câu hỏi không trùng khớp hoàn toàn (thường không nên xảy ra
    vì đề bài yêu cầu chạy cùng 1 test set cho mọi biến thể).
    """
    ids_a, ids_b = set(a.records), set(b.records)
    common = sorted(ids_a & ids_b, key=lambda x: (len(x), x))

    only_a, only_b = ids_a - ids_b, ids_b - ids_a
    if only_a or only_b:
        print(f"  [CẢNH BÁO] '{a.variant}' vs '{b.variant}': lệch tập câu hỏi — "
              f"{len(only_a)} câu chỉ có ở {a.variant}, {len(only_b)} câu chỉ có ở {b.variant}. "
              f"Chỉ so sánh trên {len(common)} câu chung.", file=sys.stderr)

    correct_a = np.array([1 if a.records[qid].correct else 0 for qid in common])
    correct_b = np.array([1 if b.records[qid].correct else 0 for qid in common])
    return common, correct_a, correct_b


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """
    McNemar's test cho dữ liệu ghép cặp (paired) nhị phân đúng/sai.
    Bảng 2x2:
                    B đúng      B sai
        A đúng        n11         n10   (b = n10: A đúng, B sai)
        A sai         n01         n00   (c = n01: A sai, B đúng)

    Dùng exact binomial test trên (b, c) khi b + c nhỏ (< 25, khuyến nghị chuẩn),
    dùng chi-square (with continuity correction) khi b + c lớn.
    """
    b = int(np.sum((correct_a == 1) & (correct_b == 0)))  # A đúng, B sai
    c = int(np.sum((correct_a == 0) & (correct_b == 1)))  # A sai, B đúng
    n_disagree = b + c

    if n_disagree == 0:
        return {
            "n_disagree": 0, "b_a_only_correct": b, "c_b_only_correct": c,
            "statistic": None, "p_value": 1.0, "method": "no_disagreement",
        }

    if n_disagree < 25:
        # Exact binomial (2-sided), tương đương McNemar exact test
        res = binomtest(min(b, c), n_disagree, p=0.5, alternative="two-sided")
        p_value = res.pvalue
        method = "exact_binomial"
        statistic = None
    else:
        # Chi-square với continuity correction (McNemar cổ điển)
        statistic = (abs(b - c) - 1) ** 2 / n_disagree
        # chi2 cdf với 1 dof
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(statistic, df=1)
        method = "chi_square_continuity_corrected"

    return {
        "n_disagree": n_disagree,
        "b_a_only_correct": b,
        "c_b_only_correct": c,
        "statistic": statistic,
        "p_value": float(p_value),
        "method": method,
    }


def bootstrap_accuracy_gain_ci(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    alpha: float = 0.05,
) -> dict:
    """
    Bootstrap CI cho accuracy của từng biến thể và cho gain = acc(b) - acc(a).
    Resample theo cặp (paired bootstrap): mỗi lần lấy mẫu lại index câu hỏi có hoàn lại,
    giữ nguyên cặp (a_i, b_i) để bảo toàn tương quan giữa 2 hệ thống trên cùng câu hỏi.
    """
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    if n == 0:
        return {"error": "no_common_questions"}

    acc_a_samples = np.empty(n_bootstrap)
    acc_b_samples = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        acc_a_samples[i] = correct_a[idx].mean()
        acc_b_samples[i] = correct_b[idx].mean()

    gain_samples = acc_b_samples - acc_a_samples

    def ci(samples: np.ndarray) -> tuple[float, float]:
        lo = float(np.percentile(samples, 100 * alpha / 2))
        hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        return lo, hi

    acc_a_lo, acc_a_hi = ci(acc_a_samples)
    acc_b_lo, acc_b_hi = ci(acc_b_samples)
    gain_lo, gain_hi = ci(gain_samples)

    return {
        "n_bootstrap": n_bootstrap,
        "n_questions": n,
        "acc_a_point": float(correct_a.mean()),
        "acc_a_ci95": [acc_a_lo, acc_a_hi],
        "acc_b_point": float(correct_b.mean()),
        "acc_b_ci95": [acc_b_lo, acc_b_hi],
        "gain_point": float(correct_b.mean() - correct_a.mean()),
        "gain_ci95": [gain_lo, gain_hi],
        "gain_significant": not (gain_lo <= 0 <= gain_hi),  # CI không chứa 0 -> có ý nghĩa
    }


def win_loss_tie(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """So từng câu: A thắng (A đúng B sai) / B thắng (B đúng A sai) / Hoà (cùng đúng hoặc cùng sai)."""
    win_a = int(np.sum((correct_a == 1) & (correct_b == 0)))
    win_b = int(np.sum((correct_a == 0) & (correct_b == 1)))
    tie_both_correct = int(np.sum((correct_a == 1) & (correct_b == 1)))
    tie_both_wrong = int(np.sum((correct_a == 0) & (correct_b == 0)))
    n = len(correct_a)
    return {
        "n_questions": n,
        "a_wins": win_a,
        "b_wins": win_b,
        "ties_both_correct": tie_both_correct,
        "ties_both_wrong": tie_both_wrong,
        "a_win_rate": win_a / n if n else None,
        "b_win_rate": win_b / n if n else None,
    }


def compute_pairwise_comparison(a: VariantData, b: VariantData, n_bootstrap: int = N_BOOTSTRAP) -> dict:
    """Gộp toàn bộ so sánh cặp: win/loss/tie + McNemar + bootstrap CI."""
    common_ids, correct_a, correct_b = aligned_correctness(a, b)
    if len(common_ids) == 0:
        return {
            "variant_a": a.variant, "variant_b": b.variant,
            "error": "Không có câu hỏi chung giữa 2 biến thể.",
        }

    wlt = win_loss_tie(correct_a, correct_b)
    mcnemar = mcnemar_test(correct_a, correct_b)
    bootstrap = bootstrap_accuracy_gain_ci(correct_a, correct_b, n_bootstrap=n_bootstrap)

    return {
        "variant_a": a.variant,
        "variant_b": b.variant,
        "n_common_questions": len(common_ids),
        "win_loss_tie": wlt,
        "mcnemar": mcnemar,
        "bootstrap": bootstrap,
    }


# ---------------------------------------------------------------------------
# Error analysis: câu hỏi mà baseline đúng nhưng biến thể mới sai (regressions)
# và ngược lại (fixes) — rất hữu ích cho phần "Phân tích lỗi" trong report.
# ---------------------------------------------------------------------------

def error_breakdown(a: VariantData, b: VariantData, max_examples: int = 20) -> dict:
    """
    a = baseline, b = biến thể đang xét.
    'fixes'      = câu a sai, b đúng  (biến thể mới sửa được lỗi của baseline)
    'regressions'= câu a đúng, b sai  (biến thể mới làm hỏng câu baseline từng đúng)
    """
    common = sorted(set(a.records) & set(b.records), key=lambda x: (len(x), x))
    fixes, regressions = [], []
    for qid in common:
        ra, rb = a.records[qid], b.records[qid]
        if not ra.correct and rb.correct:
            fixes.append(qid)
        elif ra.correct and not rb.correct:
            regressions.append(qid)

    def sample(ids: list[str], variant_data: VariantData, n: int) -> list[dict]:
        out = []
        for qid in ids[:n]:
            r = variant_data.records[qid]
            out.append({
                "question_id": qid,
                "predicted_answer": r.predicted_answer,
                "gold_answer": r.gold_answer,
                "confidence": r.confidence,
            })
        return out

    return {
        "baseline": a.variant,
        "variant": b.variant,
        "n_fixes": len(fixes),
        "n_regressions": len(regressions),
        "fixes_examples": sample(fixes, b, max_examples),
        "regressions_examples": sample(regressions, b, max_examples),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def fmt_pct(x: Optional[float]) -> str:
    return "N/A" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x*100:.2f}%"


def fmt_num(x: Optional[float], decimals: int = 1) -> str:
    return "N/A" if x is None else f"{x:,.{decimals}f}"


def print_summary_table(summaries: list[dict]) -> None:
    headers = ["Variant", "N", "Accuracy", "Invalid%", "AvgLatency(ms)", "AvgTokens/Q", "TotalCost($)"]
    rows = []
    for s in summaries:
        rows.append([
            s["variant"],
            str(s["n_total"]),
            fmt_pct(s["accuracy"]),
            fmt_pct(s["invalid_rate"]),
            fmt_num(s["avg_latency_ms"], 0),
            fmt_num(s["avg_tokens_per_question"], 0),
            fmt_num(s["total_estimated_cost_usd"], 4) if s["total_estimated_cost_usd"] is not None else "N/A",
        ])

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(r, widths)))


def write_csv(path: str, rows: list[dict], fieldnames: Optional[list[str]] = None) -> None:
    import csv
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_pairwise_for_csv(pairwise_results: list[dict]) -> list[dict]:
    flat = []
    for p in pairwise_results:
        if "error" in p:
            flat.append({"variant_a": p["variant_a"], "variant_b": p["variant_b"], "error": p["error"]})
            continue
        wlt, mc, bs = p["win_loss_tie"], p["mcnemar"], p["bootstrap"]
        flat.append({
            "variant_a": p["variant_a"],
            "variant_b": p["variant_b"],
            "n_common_questions": p["n_common_questions"],
            "a_wins": wlt["a_wins"],
            "b_wins": wlt["b_wins"],
            "ties_both_correct": wlt["ties_both_correct"],
            "ties_both_wrong": wlt["ties_both_wrong"],
            "mcnemar_p_value": mc["p_value"],
            "mcnemar_method": mc["method"],
            "mcnemar_significant_at_0.05": (mc["p_value"] is not None and mc["p_value"] < 0.05),
            "acc_a": bs.get("acc_a_point"),
            "acc_a_ci95_low": bs.get("acc_a_ci95", [None, None])[0],
            "acc_a_ci95_high": bs.get("acc_a_ci95", [None, None])[1],
            "acc_b": bs.get("acc_b_point"),
            "acc_b_ci95_low": bs.get("acc_b_ci95", [None, None])[0],
            "acc_b_ci95_high": bs.get("acc_b_ci95", [None, None])[1],
            "accuracy_gain": bs.get("gain_point"),
            "gain_ci95_low": bs.get("gain_ci95", [None, None])[0],
            "gain_ci95_high": bs.get("gain_ci95", [None, None])[1],
            "gain_significant_bootstrap": bs.get("gain_significant"),
        })
    return flat


def write_markdown_report(
    path: str,
    summaries: list[dict],
    pairwise_results: list[dict],
    error_breakdowns: list[dict],
    baseline_variant: str,
) -> None:
    lines = []
    lines.append("# Báo cáo đánh giá hệ thống Multi-Agent LLM — MedQA-USMLE\n")
    lines.append(f"Baseline dùng để so sánh: **{baseline_variant}**\n")

    lines.append("## 1. Tổng hợp theo từng biến thể\n")
    lines.append("| Variant | N | Accuracy | Invalid Rate | Avg Latency (ms) | Avg Tokens/Q | Total Cost (USD) | Pipeline |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        cost = f"{s['total_estimated_cost_usd']:.4f}" if s["total_estimated_cost_usd"] is not None else "N/A"
        lines.append(
            f"| {s['variant']} | {s['n_total']} | {fmt_pct(s['accuracy'])} | "
            f"{fmt_pct(s['invalid_rate'])} | {fmt_num(s['avg_latency_ms'],0)} | "
            f"{fmt_num(s['avg_tokens_per_question'],0)} | {cost} | {s['pipeline'] or 'N/A'} |"
        )
    lines.append("")

    lines.append("## 2. So sánh cặp với baseline (accuracy gain, McNemar, bootstrap CI)\n")
    for p in pairwise_results:
        if "error" in p:
            lines.append(f"### {p['variant_a']} vs {p['variant_b']}\n\n⚠️ {p['error']}\n")
            continue
        wlt, mc, bs = p["win_loss_tie"], p["mcnemar"], p["bootstrap"]
        lines.append(f"### {p['variant_a']} (baseline) vs {p['variant_b']}\n")
        lines.append(f"- Số câu hỏi chung: **{p['n_common_questions']}**")
        lines.append(
            f"- Accuracy: {p['variant_a']} = {fmt_pct(bs['acc_a_point'])} "
            f"(95% CI [{fmt_pct(bs['acc_a_ci95'][0])}, {fmt_pct(bs['acc_a_ci95'][1])}]), "
            f"{p['variant_b']} = {fmt_pct(bs['acc_b_point'])} "
            f"(95% CI [{fmt_pct(bs['acc_b_ci95'][0])}, {fmt_pct(bs['acc_b_ci95'][1])}])"
        )
        gain_sig = "✅ có ý nghĩa" if bs["gain_significant"] else "❌ chưa có ý nghĩa"
        lines.append(
            f"- **Accuracy gain**: {bs['gain_point']*100:+.2f} điểm % "
            f"(95% CI [{bs['gain_ci95'][0]*100:+.2f}%, {bs['gain_ci95'][1]*100:+.2f}%]) — {gain_sig} (bootstrap, α=0.05)"
        )
        lines.append(
            f"- **McNemar test** ({mc['method']}): b={mc['b_a_only_correct']} "
            f"(chỉ {p['variant_a']} đúng), c={mc['c_b_only_correct']} (chỉ {p['variant_b']} đúng), "
            f"p-value = {mc['p_value']:.4f} "
            f"({'✅ có ý nghĩa thống kê' if mc['p_value'] is not None and mc['p_value'] < 0.05 else '❌ chưa có ý nghĩa'} ở α=0.05)"
        )
        lines.append(
            f"- **Win/Loss/Tie**: {p['variant_a']} thắng {wlt['a_wins']} câu, "
            f"{p['variant_b']} thắng {wlt['b_wins']} câu, "
            f"hoà (cả 2 đúng) {wlt['ties_both_correct']} câu, hoà (cả 2 sai) {wlt['ties_both_wrong']} câu"
        )
        lines.append("")

    lines.append("## 3. Phân tích lỗi (Error Analysis)\n")
    for eb in error_breakdowns:
        lines.append(f"### {eb['baseline']} → {eb['variant']}\n")
        lines.append(f"- Số câu được **sửa đúng** (baseline sai, variant đúng): **{eb['n_fixes']}**")
        lines.append(f"- Số câu bị **hồi quy** (baseline đúng, variant sai): **{eb['n_regressions']}**")
        if eb["regressions_examples"]:
            lines.append("\n**Ví dụ câu bị hồi quy (tối đa hiển thị một phần):**\n")
            lines.append("| question_id | predicted | gold |")
            lines.append("|---|---|---|")
            for ex in eb["regressions_examples"][:10]:
                lines.append(f"| {ex['question_id']} | {ex['predicted_answer']} | {ex['gold_answer']} |")
        lines.append("")

    lines.append("---\n*Báo cáo được sinh tự động bởi `evaluate.py`.*")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá các biến thể hệ thống Multi-Agent LLM cho MedQA-USMLE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--files", nargs="*", default=[],
                         help="Danh sách file .jsonl cụ thể (hỗ trợ glob pattern).")
    parser.add_argument("--dir", default=None,
                         help="Thư mục để quét *.jsonl nếu không dùng --files (mặc định: thư mục hiện tại).")
    parser.add_argument("--baseline", default=None,
                         help="Tên variant dùng làm baseline để so sánh (mặc định: tự tìm variant có 'v0').")
    parser.add_argument("--official-n", type=int, default=None,
                         help="Số câu chính thức của test set (vd 1273 cho MedQA-USMLE full) "
                              "để tính accuracy trên mẫu số cố định thay vì số câu thực chạy được.")
    parser.add_argument("--limit-first-n", type=int, default=None,
                         help="Chỉ giữ lại N câu hỏi đầu tiên (theo thứ tự question_id tăng dần, "
                              "vd test_00000..test_00199 cho N=200) ở MỌI biến thể trước khi tính "
                              "toán bất kỳ metric nào. Dùng khi các file có số câu khác nhau và bạn "
                              "muốn ép so sánh công bằng trên cùng một tập câu hỏi con.")
    parser.add_argument("--out-dir", default=".",
                         help="Thư mục lưu output CSV/JSON/Markdown (mặc định thư mục hiện tại).")
    parser.add_argument("--report", default="report.md",
                         help="Tên file Markdown report (đặt rỗng '' để bỏ qua).")
    parser.add_argument("--bootstrap-n", type=int, default=N_BOOTSTRAP,
                         help=f"Số lần resample bootstrap (mặc định {N_BOOTSTRAP}).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Nạp file
    filepaths = discover_files(args.files, args.dir)
    print(f"Tìm thấy {len(filepaths)} file:")
    for fp in filepaths:
        print(f"  - {fp}")
    print()

    variants: dict[str, VariantData] = {}
    for fp in filepaths:
        vdata = load_jsonl_variant(fp)
        if vdata.variant in variants:
            print(f"  [CẢNH BÁO] Tên variant '{vdata.variant}' bị trùng giữa nhiều file "
                  f"— file sau ({fp}) sẽ ghi đè.", file=sys.stderr)
        variants[vdata.variant] = vdata
        print(f"Đã nạp variant '{vdata.variant}': {len(vdata)} câu hỏi (từ {fp})")
    print()

    if args.limit_first_n:
        print(f"==> Áp dụng --limit-first-n={args.limit_first_n}: giới hạn mọi biến thể "
              f"về cùng {args.limit_first_n} câu hỏi đầu tiên (theo question_id).\n")
        limit_to_first_n_questions(variants, args.limit_first_n)
        print()

    # 2. Xác định baseline
    baseline_name = args.baseline
    if baseline_name is None:
        candidates = [v for v in variants if "v0" in v.lower()]
        baseline_name = candidates[0] if candidates else next(iter(variants))
    if baseline_name not in variants:
        print(f"[LỖI] Không tìm thấy variant baseline '{baseline_name}' trong dữ liệu đã nạp.",
              file=sys.stderr)
        sys.exit(1)
    print(f"==> Baseline được chọn: '{baseline_name}'\n")

    # 3. Summary từng biến thể
    summaries = [
        compute_variant_summary(v, official_n=args.official_n)
        for v in variants.values()
    ]
    summaries.sort(key=lambda s: (s["variant"] != baseline_name, s["variant"]))

    print("=" * 100)
    print("TỔNG HỢP METRICS THEO TỪNG BIẾN THỂ")
    print("=" * 100)
    print_summary_table(summaries)
    print()

    write_csv(os.path.join(args.out_dir, "summary_metrics.csv"), summaries)
    print(f"Đã lưu: {os.path.join(args.out_dir, 'summary_metrics.csv')}")

    # 4. So sánh cặp: baseline vs từng biến thể còn lại
    pairwise_results = []
    error_breakdowns = []
    baseline_vdata = variants[baseline_name]

    print()
    print("=" * 100)
    print(f"SO SÁNH CẶP: {baseline_name} (baseline) vs các biến thể khác")
    print("=" * 100)

    for name, vdata in variants.items():
        if name == baseline_name:
            continue
        result = compute_pairwise_comparison(baseline_vdata, vdata, n_bootstrap=args.bootstrap_n)
        pairwise_results.append(result)

        if "error" not in result:
            wlt, mc, bs = result["win_loss_tie"], result["mcnemar"], result["bootstrap"]
            sig_mark = "***" if mc["p_value"] is not None and mc["p_value"] < 0.05 else ""
            print(f"\n{baseline_name} vs {name}  (n={result['n_common_questions']})")
            print(f"  Accuracy gain: {bs['gain_point']*100:+.2f}%  "
                  f"95% CI [{bs['gain_ci95'][0]*100:+.2f}%, {bs['gain_ci95'][1]*100:+.2f}%]")
            print(f"  McNemar p-value: {mc['p_value']:.4f} {sig_mark}  "
                  f"(b={mc['b_a_only_correct']}, c={mc['c_b_only_correct']}, method={mc['method']})")
            print(f"  Win/Loss/Tie: {name} thắng {wlt['b_wins']}, {baseline_name} thắng {wlt['a_wins']}, "
                  f"hoà đúng {wlt['ties_both_correct']}, hoà sai {wlt['ties_both_wrong']}")
        else:
            print(f"\n{baseline_name} vs {name}: {result['error']}")

        eb = error_breakdown(baseline_vdata, vdata)
        error_breakdowns.append(eb)

    write_csv(
        os.path.join(args.out_dir, "pairwise_comparison.csv"),
        flatten_pairwise_for_csv(pairwise_results),
    )
    print(f"\nĐã lưu: {os.path.join(args.out_dir, 'pairwise_comparison.csv')}")

    # 5. Markdown report
    if args.report:
        report_path = os.path.join(args.out_dir, args.report)
        write_markdown_report(report_path, summaries, pairwise_results, error_breakdowns, baseline_name)
        print(f"Đã lưu: {report_path}")

    # 6. Lưu toàn bộ kết quả raw ra JSON (hữu ích để debug / phân tích thêm)
    json_path = os.path.join(args.out_dir, "full_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": baseline_name,
            "summaries": summaries,
            "pairwise": pairwise_results,
            "error_breakdowns": error_breakdowns,
        }, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu: {json_path}")


if __name__ == "__main__":
    main()
