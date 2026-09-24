import tempfile
import unittest
from pathlib import Path

from parse_harness.render import discover_pdfs, render_scale


class RenderTests(unittest.TestCase):
    def test_discover_pdfs_accepts_file_or_directory_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.PDF").write_bytes(b"pdf")
            (root / "a.pdf").write_bytes(b"pdf")
            (root / "note.txt").write_text("ignore")

            self.assertEqual([path.name for path in discover_pdfs(root)], ["a.pdf", "b.PDF"])
            self.assertEqual(discover_pdfs(root / "a.pdf"), [(root / "a.pdf").resolve()])

    def test_render_scale_uses_parse_2_maximum_canvas(self):
        portrait_scale = render_scale(612, 792)
        landscape_scale = render_scale(792, 612)

        self.assertAlmostEqual(portrait_scale, 2048 / 792)
        self.assertAlmostEqual(landscape_scale, 1664 / 792)
        self.assertGreater(612 * portrait_scale, 1024)
        self.assertGreater(612 * landscape_scale, 1280)


if __name__ == "__main__":
    unittest.main()
