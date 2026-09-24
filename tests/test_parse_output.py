import unittest

from parse_harness.parse_output import extract_blocks, model_to_source_bbox


class ParseOutputTests(unittest.TestCase):
    def test_extract_blocks_accepts_parse_2_chart_class(self):
        raw = "<x_0.1><y_0.2>Quarterly revenue<x_0.9><y_0.8><class_Chart>"

        blocks = extract_blocks(raw, source_size=(1664, 2048))

        self.assertEqual(blocks[0]["category"], "Chart")
        self.assertEqual(blocks[0]["text"], "Quarterly revenue")

    def test_extract_blocks_preserves_generation_order_and_coordinates(self):
        raw = (
            "<x_0.1><y_0.2>First<tbc><x_0.4><y_0.5><class_Text>"
            "<x_0.2><y_0.3>Second<x_0.8><y_0.9><class_Title>"
        )

        blocks = extract_blocks(raw, model_size=(1000, 2000), source_size=(1000, 2000))

        self.assertEqual([block["text"] for block in blocks], ["First", "Second"])
        self.assertEqual(blocks[0]["normalized_bbox"], [0.1, 0.2, 0.4, 0.5])
        self.assertEqual(blocks[0]["model_bbox"], [100, 400, 400, 1000])
        self.assertEqual(blocks[0]["source_bbox"], [100, 400, 400, 1000])

    def test_small_source_image_removes_center_padding_without_upscaling(self):
        mapped = model_to_source_bbox(
            (564.0, 960.0, 1097.0, 1082.0),
            source_size=(539, 125),
            model_size=(1664, 2048),
        )

        self.assertEqual(mapped, (2, 0, 535, 121))


if __name__ == "__main__":
    unittest.main()
