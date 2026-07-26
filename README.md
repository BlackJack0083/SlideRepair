# SlideRepair

Reference implementation of **SlideGuard** and evaluation code for the
**SlideRepair** benchmark.

> **Full dataset:** [Download SlideRepair from Google Drive](https://drive.google.com/file/d/1vtD6zpVGROz3r3_75uSBntNpKFWHP9yF/view?usp=drive_link)

SlideRepair evaluates factual error detection and repair in data-driven
presentation slides. The benchmark contains 3,905 clean slides and 25,826
corrupted cases with Scope, Value, and Claim errors. This repository contains
the method and evaluation code, together with a small example benchmark drawn
from the train, validation, and test splits.

## Overview

SlideGuard processes each slide in three stages:

1. **Slide understanding and analytical state extraction** parses editable
   PPTX elements, labels their roles, and extracts the data source and
   computation logic.
2. **Evidence-grounded state validation** checks the analytical scope against
   PostgreSQL data, recomputes chart or table content, detects factual errors,
   and obtains client clarification when required.
3. **Verified-state slide repair** writes verified data and text back to the
   affected native PPTX elements while preserving unaffected content and
   layout.

The workflow outputs a repaired PPTX, detected issues, stage logs, tool calls,
intermediate analytical states, and evaluation metrics.

## Repository Layout

```text
method/
  agents/                  Slide parser, analysis, validation, and repair agents
  eval/                    Detection, stage, artifact, and aggregate metrics
  pipeline/                End-to-end SlideGuard workflow
  prompts/                 Agent prompts
  utils/                   PPTX, JSON, client, and dataframe utilities
scripts/
  run_slide_review_eval.py
  analyze_slide_review_results.py
examples/benchmark/        9 clean slides and 61 corrupted cases
```

Dataset construction, corruption generation, and oracle-output ablation code
are intentionally not included in this release.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/BlackJack0083/SlideRepair.git
cd SlideRepair
uv sync
```

Create a `.env` file from `.env.example` and provide:

- PostgreSQL connection variables: `SQL_USER`, `SQL_PASSWORD`, `SQL_HOST`,
  `SQL_PORT`, and `SQL_DB`.
- An OpenAI-compatible model endpoint: `DASHSCOPE_API_KEY`,
  `DASHSCOPE_MODEL`, and `DASHSCOPE_BASE_URL`.
- Optional stage-specific endpoints using the `PARSER_`, `ANALYSIS_`,
  `DATA_SOURCE_`, `CONTENT_`, and `CLIENT_` prefixes. The parser model must
  support image input.

The evidence-grounded validation stage requires access to the PostgreSQL
database associated with SlideRepair.

## Run the Example Evaluation

Evaluate corrupted cases with fixed client responses:

```bash
uv run python scripts/run_slide_review_eval.py \
  --benchmark-root examples/benchmark \
  --split test \
  --case-type corrupted \
  --limit 5 \
  --workers 2 \
  --client-mode deterministic \
  --output-dir output/example_test
```

Evaluate clean slides:

```bash
uv run python scripts/run_slide_review_eval.py \
  --benchmark-root examples/benchmark \
  --split test \
  --case-type clean \
  --limit 3 \
  --workers 2 \
  --client-mode deterministic \
  --output-dir output/example_clean
```

Use `--client-mode llm` to paraphrase the case-specific client feedback with
the configured client model. Each case trace is written as JSON, and repaired
artifacts are stored under the run's `work/` directory.

## Aggregate Results

```bash
uv run python scripts/analyze_slide_review_results.py \
  --result-dir output/example_test \
  --model your-model-name \
  --client-mode deterministic \
  --output-json output/example_test_summary.json \
  --output-csv output/example_test_summary.csv
```

The evaluator reports error-category and fine-grained issue detection,
stage-level accuracy, content repair accuracy, and end-to-end slide repair
success rate.
