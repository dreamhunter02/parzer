import tempfile
import unittest
from pathlib import Path

from PIL import Image

from parse_harness.inference import build_message_content, generation_options, infer_page, run_inference
from parse_harness.render import PageItem


class InferenceTests(unittest.TestCase):
    def test_parse_2_message_includes_control_prompt_before_image(self):
        image_url = "data:image/png;base64,abc"
        prompt = "<control>"

        self.assertEqual(
            build_message_content(image_url=image_url, prompt=prompt),
            [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        )

    def test_parse_2_generation_options_preserve_structure_tokens(self):
        self.assertEqual(
            generation_options(),
            {"repetition_penalty": 1.1, "top_k": 1, "skip_special_tokens": False},
        )

    def make_page(self, root: Path) -> PageItem:
        image_path = root / "page.png"
        Image.new("RGB", (100, 200), "white").save(image_path)
        return PageItem(
            page_id="doc:page_0001",
            document_id="doc",
            source_pdf="/input/doc.pdf",
            page_number=1,
            image_path=str(image_path),
            image_width=100,
            image_height=200,
            render_scale=1.0,
        )

    def test_infer_page_returns_provenance_raw_generation_and_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = self.make_page(Path(tmp))
            raw = "<x_0.1><y_0.2>Hello<x_0.9><y_0.8><class_Text>"

            record = infer_page(page, lambda _path: raw)

            self.assertEqual(record["page_id"], page.page_id)
            self.assertEqual(record["raw_generation"], raw)
            self.assertEqual(record["blocks"][0]["text"], "Hello")
            self.assertEqual(record["blocks"][0]["normalized_bbox"], [0.1, 0.2, 0.9, 0.8])

    def test_run_inference_retries_then_resume_uses_cached_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page = self.make_page(root)
            attempts = 0

            def flaky(_path: Path) -> str:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("temporary")
                return "<x_0><y_0>ok<x_1><y_1><class_Text>"

            first = run_inference([page], root, flaky, concurrency=1, retries=1, resume=False)
            second = run_inference(
                [page],
                root,
                lambda _path: self.fail("resume should not call the endpoint"),
                concurrency=1,
                retries=1,
                resume=True,
            )

            self.assertEqual(attempts, 2)
            self.assertEqual(first, second)
            self.assertTrue((root / "raw_responses/doc/page_0001.json").exists())


if __name__ == "__main__":
    unittest.main()
