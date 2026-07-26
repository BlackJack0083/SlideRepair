from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from method.eval.result_analysis import (  # noqa: E402
    analyze_result_directories,
    write_analysis_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate slide-review case traces into paper metrics."
    )
    parser.add_argument("--result-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--client-mode",
        choices=("deterministic", "llm"),
        required=True,
    )
    parser.add_argument("--configuration", default="full")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    analysis = analyze_result_directories(args.result_dir)
    write_analysis_csv(
        analysis,
        args.output_csv,
        model=args.model,
        client_mode=args.client_mode,
        configuration=args.configuration,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
