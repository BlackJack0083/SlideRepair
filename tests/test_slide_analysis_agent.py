from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any

from method.agents.slide_analysis.agent import SlideAnalysisAgent


class BadFunctionLogicClient:
    async def achat(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> str:
        del system_prompt, user_prompt, kwargs
        return json.dumps(
            {
                "table_type": "field-constraint",
                "dimensions": [
                    {
                        "source_col": "category",
                        "target_col": "year",
                        "method": "period",
                        "time_granularity": "year",
                    }
                ],
                "metrics": [
                    {
                        "name": "trade_counts",
                        "source_col": "trade_sets",
                        "agg_func": "count",
                        "filter_condition": {"trade_sets": 1},
                    }
                ],
            }
        )


class SlideAnalysisAgentTest(unittest.TestCase):
    def test_function_logic_rejects_visible_category_as_source_col(self) -> None:
        agent = SlideAnalysisAgent(client=BadFunctionLogicClient())

        with self.assertRaisesRegex(ValueError, "dimension source_col"):
            asyncio.run(
                agent._extract_function_logic(
                    {
                        "caption": "Beijing Liangxiang Transaction Count (2020-2021)",
                        "table_data": [
                            {"category": "2020", "trade_counts": "10"},
                            {"category": "2021", "trade_counts": "12"},
                        ],
                    },
                    0,
                )
            )


if __name__ == "__main__":
    unittest.main()
