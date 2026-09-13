"""Script chạy tự động toàn bộ suite benchmark (V1 -> V3-qr) trên 200 câu test set.
Đo chính xác thời gian thực thi (wall-clock time), accuracy và xuất báo cáo so sánh.
"""

import sys
import time
import json
import subprocess
from pathlib import Path

# Fix Windows encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from core.config import load_config
from modes.benchmark import run_benchmark

VARIANTS = [
    ("v2-qr", "configs/v2-qr.yaml"),
    ("v3-qr", "configs/v3-qr.yaml"),
]

# Baseline V0 & V1 đã chạy xong 200/200 câu trước đó
V0_FILE = "output/predictions_v0_1789294527.jsonl"
V0_TIME_STR = "3m 04s"
V0_ACC = "89.0%"

V1_FILE = "output/predictions_v1_1789296941.jsonl"
V1_TIME_STR = "3m 58s"
V1_ACC = "82.5%"

def main():
    suite_start = time.time()
    results = []

    print("=" * 60)
    print("MED-AI BENCHMARK SUITE — 200 CÂU TEST SET (8 WORKERS)")
    print("Mô hình: Reasoning=deepseek-v4-flash, Embedding=baai/bge-m3")
    print("=" * 60)

    for var_name, cfg_path in VARIANTS:
        print(f"\n{'#' * 60}")
        print(f"# BẮT ĐẦU CHẠY: {var_name.upper()} (split=test, limit=200, workers=8)")
        print(f"{'#' * 60}\n")
        
        t0 = time.time()
        try:
            cfg = load_config(cfg_path)
            res = run_benchmark(
                config=cfg,
                split="test",
                limit=200,
                output_path=None,
                workers=8,
            )
            elapsed = time.time() - t0
            res["elapsed_sec"] = elapsed
            res["elapsed_str"] = f"{int(elapsed // 60)}m {int(elapsed % 60):02d}s"
            results.append(res)
            print(f"\n>>> HOÀN THÀNH {var_name}: {res['correct']}/{res['total']} ({res['accuracy']:.1%}) trong {res['elapsed_str']}")
        except Exception as exc:
            print(f"\n[LỖI] Biến thể {var_name} thất bại: {exc}", file=sys.stderr)

    total_suite_time = time.time() - suite_start
    total_min = int(total_suite_time // 60)
    total_sec = int(total_suite_time % 60)

    print("\n" + "=" * 60)
    print("TỔNG HỢP KẾT QUẢ THỰC THI (8 WORKERS PARALLEL)")
    print("=" * 60)
    print(f"{'Variant':<10} | {'Total':<6} | {'Correct':<8} | {'Accuracy':<10} | {'Thời gian chạy':<15} | Output File")
    print("-" * 80)
    print(f"{'v0':<10} | {'200':<6} | {'178':<8} | {V0_ACC:<10} | {V0_TIME_STR:<15} | {V0_FILE}")
    print(f"{'v1':<10} | {'200':<6} | {'165':<8} | {V1_ACC:<10} | {V1_TIME_STR:<15} | {V1_FILE}")
    for r in results:
        print(f"{r['variant']:<10} | {r['total']:<6} | {r['correct']:<8} | {r['accuracy']:.1%}{'':<4} | {r['elapsed_str']:<15} | {r['output_file']}")
    print("-" * 80)
    print(f"Tổng thời gian chạy đợt này: {total_min}m {total_sec:02d}s\n")

    # Chạy evaluate.py để tính thống kê và tạo báo cáo
    files_to_eval = [V0_FILE, V1_FILE] + [r["output_file"] for r in results if "output_file" in r]
    print(f"Đang chạy evaluate.py trên {len(files_to_eval)} files...")
    cmd = [sys.executable, "../evaluate.py", "--files"] + files_to_eval + ["--report", "BENCHMARK_REPORT_OPTIMIZED.md"]
    try:
        subprocess.run(cmd, check=True)
        print("\nĐã tạo thành công báo cáo chi tiết: BENCHMARK_REPORT_OPTIMIZED.md")
    except Exception as e:
        print(f"Lỗi khi chạy evaluate.py: {e}")

if __name__ == "__main__":
    main()

