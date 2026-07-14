"""Entry point - dispatcher chọn Mode 1 (benchmark) hoặc Mode 2 (chat).

Import mode được chọn theo kiểu "lazy" (import bên trong nhánh if/elif),
không import cả 2 mode ở đầu file. Nhờ vậy nếu xoá modes/chat.py,
`python main.py --mode benchmark` vẫn chạy bình thường.
"""

import argparse

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Med-AI")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "chat"],
        required=True,
        help="benchmark: sinh prediction trên MedQA-USMLE-4-options | chat: hỏi đáp tự do",
    )
    parser.add_argument(
        "--config",
        help="Đường dẫn file config YAML (bắt buộc cho --mode benchmark, VD configs/v0.yaml)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split dataset cho benchmark: train hoặc test (mặc định: test, 1273 câu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số câu hỏi khi benchmark (VD --limit 50 để debug nhanh trên train)",
    )
    args = parser.parse_args()

    if args.mode == "benchmark":
        if not args.config:
            parser.error("--config là bắt buộc khi --mode benchmark (VD --config configs/v0.yaml)")
        from modes.benchmark import run_benchmark

        run_benchmark(config_path=args.config, split=args.split, limit=args.limit)

    elif args.mode == "chat":
        from modes.chat import run_chat

        run_chat()


if __name__ == "__main__":
    main()
