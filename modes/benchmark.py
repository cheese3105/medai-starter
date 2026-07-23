"""Mode 1 - Benchmark.

Sinh prediction cho MedQA-USMLE-4-options, lưu ra
predictions_{variant}_{split}.jsonl.

Nguyên tắc bắt buộc:
1. KHÔNG tự tính accuracy/metric ở đây - việc chấm điểm giao hết cho
   evaluate.py (riêng, join predictions với gold_{split}.jsonl theo
   question_id).
2. KHÔNG đưa gold answer (answer_idx) vào bất kỳ đâu trong pipeline sinh
   câu trả lời - gold answer chỉ ghi riêng ra gold_{split}.jsonl.
3. Schema predictions.jsonl giống hệt nhau cho mọi variant (V0-V4) - các
   field nếu chưa dùng (retrieved_docs, agent_trace) vẫn có mặt, giá trị
   null.
4. dataset GBaker/MedQA-USMLE-4-options chỉ có 2 split (train, test,
   không có "dev") - dùng split="train" kèm --limit để debug nhanh, chỉ
   chạy split="test" đầy đủ 1 lần khi config đã khoá.
"""

import json
import time
from pathlib import Path

from datasets import load_dataset

from graph import build_graph
from run_config import load_run_config

LABELS = ["A", "B", "C", "D"]


def _to_agent_input(row: dict, question_id: str) -> dict:
    """Adapter: dataset -> AgentState. KHÔNG đưa answer_idx vào đây."""
    return {
        "question_id": question_id,
        "question": row["question"],
        "choices": [row["options"][label] for label in LABELS],
        "answer": None,
        "explanation": None,
        "confidence": None,
        "raw_output": None,
        "latency_ms": None,
        "token_usage": None,
        "estimated_cost": None,
        "retrieved_docs": None,
        "agent_trace": None,
    }


def run_benchmark(config_path: str, split: str = "test", limit: int | None = None) -> None:
    run_config = load_run_config(config_path)

    print(f"Variant: {run_config.variant} | Model: {run_config.model} | Split: {split}")

    if split == "test" and limit is None:
        print(
            "⚠️  Đang chạy FULL test set (1273 câu) - đảm bảo config đã "
            "\"khoá\" trước khi chạy chính thức (nguyên tắc #4)."
        )

    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    print(f"Số câu hỏi: {len(dataset)}")

    app = build_graph(run_config)

    timestamp = int(time.time())
    run_dir_name = f"{run_config.variant}_{split}_{timestamp}"
    run_dir = Path("output") / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pred_path = run_dir / f"predictions_{run_config.variant}_{split}_{timestamp}.jsonl"
    gold_path = run_dir / f"gold_{split}_{timestamp}.jsonl"
    config_out_path = run_dir / f"run_config_{run_config.variant}_{split}_{timestamp}.jsonl"
    gold_already_exists = gold_path.exists()

    # run_config lưu 1 lần/file riêng (dạng jsonl)
    with open(config_out_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {**run_config.to_dict(), "split": split, "num_questions": len(dataset)},
                ensure_ascii=False,
            )
            + "\n"
        )

    with open(pred_path, "w", encoding="utf-8") as pred_f, \
            open(gold_path, "a" if gold_already_exists else "w", encoding="utf-8") as gold_f:

        for i, row in enumerate(dataset):
            question_id = f"{split}_{i}"
            agent_input = _to_agent_input(row, question_id)

            try:
                out = app.invoke(agent_input)
            except Exception as e:  # 1 câu lỗi không được làm sập cả lượt chạy
                out = {**agent_input, "answer": "INVALID", "raw_output": f"ERROR: {e}"}

            prediction = {
                "question_id": question_id,
                "question": out.get("question"),
                "choices": out.get("choices"),
                "predicted_answer": out.get("answer"),
                "explanation": out.get("explanation"),
                "raw_output": out.get("raw_output"),
                "latency_ms": out.get("latency_ms"),
                "token_usage": out.get("token_usage"),
                "estimated_cost": out.get("estimated_cost"),
                "retrieved_docs": out.get("retrieved_docs"),
                "agent_trace": out.get("agent_trace"),
            }
            pred_f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

            # Gold answer không phụ thuộc variant -> chỉ cần ghi 1 lần cho mỗi split
            if not gold_already_exists:
                gold_f.write(
                    json.dumps(
                        {"question_id": question_id, "gold_answer": row["answer_idx"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            print(f"[{i + 1}/{len(dataset)}] {question_id} -> {prediction['predicted_answer']}")

    print(f"\nĐã lưu predictions: {pred_path}")
    print(f"Đã lưu run_config:  {config_out_path}")
    if not gold_already_exists:
        print(f"Đã lưu gold answers: {gold_path}")
