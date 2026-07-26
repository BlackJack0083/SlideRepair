from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from method.agents import SlideParserAgent, SlideReviewInput, extract_pptx_elements


class SequentialRoleClient:
    async def achat(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path,
        response_format: str,
    ) -> str:
        del system_prompt, image_path, response_format
        payload = json.loads(user_prompt.split("Input elements:\n", 1)[1])
        roles = ["title", "summary", "caption", "chart-bar"]
        return json.dumps(
            {
                "roles": [
                    {"id": str(element["id"]), "role": roles[idx]}
                    for idx, element in enumerate(payload["elements"])
                ]
            }
        )


class TestSlideParser(unittest.TestCase):
    def setUp(self) -> None:
        source_dir = Path(
            "examples/benchmark/split/test/s_000c4d3e0165879e/gt"
        )
        self._tmpdir = tempfile.TemporaryDirectory()
        self.sample_dir = Path(self._tmpdir.name)
        shutil.copyfile(source_dir / "slide.pptx", self.sample_dir / "slide.pptx")
        self.pptx_path = self.sample_dir / "slide.pptx"
        self.image_path = self.sample_dir / "slide.png"
        self.image_path.write_bytes(b"")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_extract_pptx_elements_keeps_template_slide_shape_without_roles(self):
        observed_slide = extract_pptx_elements(self.pptx_path)

        self.assertIn("slide_size", observed_slide)
        self.assertEqual(len(observed_slide["elements"]), 4)
        self.assertEqual(
            [element["type"] for element in observed_slide["elements"]],
            ["textBox", "textBox", "textBox", "chart"],
        )
        self.assertTrue(
            all("role" not in element for element in observed_slide["elements"])
        )
        self.assertNotIn("data", observed_slide["elements"][-1])

    def test_slide_parser_agent_merges_roles_and_exports_csv(self):
        result = asyncio.run(
            SlideParserAgent(client=SequentialRoleClient()).arun(
                SlideReviewInput(
                    pptx_path=self.pptx_path,
                    image_path=self.image_path,
                )
            )
        )

        elements = result["observed_slide"]["elements"]
        self.assertEqual(
            [element["role"] for element in elements],
            ["title", "summary", "caption", "chart-bar"],
        )
        self.assertNotIn("role_assignments", result)
        self.assertNotIn("case_id", result["ppt_representation"])
        self.assertNotIn("data", elements[-1])
        self.assertNotIn("shape_kind", elements[-1])
        table = result["ppt_representation"]["structured_tables"][0]
        self.assertEqual(table["body"]["data_path"], "data.csv")
        case_dir = Path(result["ppt_representation"]["case_dir"])
        self.assertTrue((case_dir / table["body"]["data_path"]).exists())


if __name__ == "__main__":
    unittest.main()
