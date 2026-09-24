import tempfile
import unittest
from pathlib import Path

from PIL import Image

from parse_harness.assembly import write_document_outputs


class AssemblyTests(unittest.TestCase):
    def test_zoom_output_preserves_order_and_replaces_figure_with_resolving_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page_path = root / "page.png"
            Image.new("RGB", (100, 100), "white").save(page_path)
            records = [
                {
                    "document_id": "sample",
                    "page_number": 1,
                    "image_path": str(page_path),
                    "blocks": [
                        {"category": "Title", "text": "# Real title", "source_bbox": [0, 0, 100, 10]},
                        {"category": "Picture", "text": "incorrect description", "source_bbox": [10, 20, 50, 60]},
                        {"category": "Text", "text": "After figure", "source_bbox": [0, 70, 100, 90]},
                    ],
                }
            ]

            paths = write_document_outputs(records, root, output_mode="both", figure_mode="crop")

            raw = paths["raw"].read_text()
            zoom = paths["zoom"].read_text()
            self.assertIn("incorrect description", raw)
            self.assertEqual(
                zoom,
                "# Real title\n\n![Picture](../../assets/sample/page_0001_picture_01.png)\n\nAfter figure\n",
            )
            self.assertNotIn("# Extracted Content", zoom)
            self.assertNotIn("## Page", zoom)
            crop_path = root / "assets/sample/page_0001_picture_01.png"
            self.assertTrue(crop_path.exists())
            with Image.open(crop_path) as crop:
                self.assertEqual(crop.size, (40, 40))

    def test_pages_are_concatenated_in_numeric_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {"document_id": "sample", "page_number": 2, "blocks": [{"category": "Text", "text": "second"}]},
                {"document_id": "sample", "page_number": 1, "blocks": [{"category": "Text", "text": "first"}]},
            ]

            paths = write_document_outputs(records, root, output_mode="raw", figure_mode="description")

            self.assertEqual(paths["raw"].read_text(), "first\n\nsecond\n")

    def test_invalid_second_figure_crop_falls_back_to_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page_path = root / "page.png"
            Image.new("RGB", (100, 100), "white").save(page_path)
            records = [
                {
                    "document_id": "sample",
                    "page_number": 1,
                    "image_path": str(page_path),
                    "blocks": [
                        {"category": "Picture", "text": "first", "source_bbox": [0, 0, 20, 20]},
                        {"category": "Figure", "text": "fallback", "source_bbox": [50, 50, 50, 50]},
                    ],
                }
            ]

            paths = write_document_outputs(records, root, output_mode="zoom", figure_mode="crop")

            self.assertIn("fallback", paths["zoom"].read_text())


if __name__ == "__main__":
    unittest.main()
