"""Script chạy tự động toàn bộ suite benchmark (V0 -> V3-qr) trên MedQA-USMLE test set.
Đo chính xác thời gian thực thi (wall-clock time), accuracy và xuất báo cáo so sánh.
"""

import sys
import time
import subprocess
import argparse
from pathlib import Path

# Fix Windows encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from core.config import load_config
from modes.benchmark import run_benchmark

VARIANTS = [
    ("v0", "configs/v0.yaml"),
    ("v1", "configs/v1.yaml"),
    ("v2-qr", "configs/v2-qr.yaml"),
    ("v3-qr", "configs/v3-qr.yaml"),
]

def main():
    parser = argparse.ArgumentParser(description="Chạy tự động toàn bộ Benchmark Suite (V0 -> V3-QR)")
    parser.add_argument("--limit", type=int, default=200, help="Số câu hỏi chạy cho mỗi biến thể (mặc định: 200)")
    parser.add_argument("--workers", type=int, default=8, help="Số worker chạy song song (mặc định: 8)")
    parser.add_argument("--split", default="test", choices=["test", "train", "dev"], help="Dataset split (mặc định: test)")
    parser.add_argument("--report", default="BENCHMARK_REPORT_OPTIMIZED.md", help="Tên file markdown xuất báo cáo")
    args = parser.parse_args()

    suite_start = time.time()
    results = []

    print("=" * 60)
    print(f"MED-AI BENCHMARK SUITE — {args.limit} CÂU ({args.split.upper()} SET)")
    print(f"Cấu hình: workers={args.workers}, split={args.split}")
    print("=" * 60)

    for var_name, cfg_path in VARIANTS:
        print(f"\n{'#' * 60}")
        print(f"# BẮT ĐẦU CHẠY: {var_name.upper()} (split={args.split}, limit={args.limit}, workers={args.workers})")
        print(f"{'#' * 60}\n")
        
        t0 = time.time()
        try:
            cfg = load_config(cfg_path)
            res = run_benchmark(
                config=cfg,
                split=args.split,
                limit=args.limit,
                output_path=None,
                workers=args.workers,
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
    for r in results:
        print(f"{r['variant']:<10} | {r['total']:<6} | {r['correct']:<8} | {r['accuracy']:.1%}{'':<4} | {r['elapsed_str']:<15} | {r['output_file']}")
    print("-" * 80)
    print(f"Tổng thời gian toàn bộ suite: {total_min}m {total_sec:02d}s\n")

    # Chạy evaluate.py để tính thống kê và tạo báo cáo
    files_to_eval = [r["output_file"] for r in results if "output_file" in r and r["output_file"]]
    if files_to_eval:
        print(f"Đang chạy evaluate.py trên {len(files_to_eval)} files...")
        eval_script = Path("evaluate/evaluate.py")
        if not eval_script.exists():
            eval_script = Path("../evaluate.py")
        
        cmd = [sys.executable, str(eval_script), "--files"] + files_to_eval + ["--report", args.report]
        try:
            subprocess.run(cmd, check=True)
            print(f"\nĐã tạo thành công báo cáo chi tiết: {args.report}")
        except Exception as e:
            print(f"Lỗi khi chạy evaluate.py: {e}")
    else:
        print("Không có file kết quả hợp lệ để đánh giá.")

if __name__ == "__main__":
    main()
