"""MED-AI — CLI entrypoint.

Usage:
  python main.py --mode benchmark --config configs/v0.yaml --split test --limit 20
  python main.py --mode chat --config configs/v3-chat.yaml
"""

import argparse
import sys

from core.config import load_config


def main():
    parser = argparse.ArgumentParser(description="MED-AI System")
    parser.add_argument("--mode", choices=["benchmark", "chat"],
                        required=True, help="Chế độ chạy")
    parser.add_argument("--config", required=True,
                        help="Đường dẫn file config YAML")
    parser.add_argument("--split", default="test",
                        help="Dataset split (benchmark only): train/test")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số câu hỏi (benchmark only)")
    parser.add_argument("--output", default=None,
                        help="Đường dẫn file output JSONL (benchmark only)")

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Lỗi load config: {e}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "benchmark":
        from modes.benchmark import run_benchmark
        run_benchmark(
            config=config,
            split=args.split,
            limit=args.limit,
            output_path=args.output,
        )
    elif args.mode == "chat":
        from modes.chat import run_chat
        run_chat(config=config)


if __name__ == "__main__":
    main()
