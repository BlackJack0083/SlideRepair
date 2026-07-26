from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from method.eval import SlideReviewEvalConfig, run_slide_review_eval  # noqa: E402
from method.eval.slide_review_eval import to_jsonable  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the slide review workflow on benchmark cases."
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--case-type",
        choices=("corrupted", "clean"),
        default="corrupted",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--case-timeout-sec", type=int, default=900)
    parser.add_argument("--case-retries", type=int, default=2)
    parser.add_argument("--retry-base-sleep-sec", type=float, default=8.0)
    parser.add_argument(
        "--client-mode",
        choices=("deterministic", "llm"),
        default="deterministic",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/slide_review_eval"))
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    summary = await run_slide_review_eval(
        SlideReviewEvalConfig(
            benchmark_root=args.benchmark_root,
            split=args.split,
            case_type=args.case_type,
            limit=args.limit,
            workers=args.workers,
            case_timeout_sec=args.case_timeout_sec,
            case_retries=args.case_retries,
            retry_base_sleep_sec=args.retry_base_sleep_sec,
            client_mode=args.client_mode,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
