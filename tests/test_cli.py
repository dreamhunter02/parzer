import unittest

from parse_harness.cli import build_parser


class CliTests(unittest.TestCase):
    def test_extract_defaults_match_documented_interface(self):
        args = build_parser().parse_args(
            [
                "extract",
                "--input",
                "PDFs",
                "--output",
                "results",
                "--base-url",
                "http://127.0.0.1:8000/v1",
            ]
        )

        self.assertEqual(args.model, "nvidia/NVIDIA-Nemotron-Parse-2.0")
        self.assertEqual(args.output_mode, "both")
        self.assertEqual(args.figure_mode, "crop")
        self.assertEqual(args.concurrency, 8)
        self.assertEqual(args.max_tokens, 9000)
        self.assertEqual(args.retries, 2)


if __name__ == "__main__":
    unittest.main()
