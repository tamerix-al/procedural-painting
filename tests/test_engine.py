import unittest
import numpy as np
from painting.engine import BUILTIN_STYLES, apply_stroke, build_painting


class TestPainting(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((40, 60, 3), dtype=np.uint8)
        self.image[:, :30] = [190, 60, 40]
        self.image[:, 30:] = [30, 110, 190]

    def test_all_styles(self):
        for style in BUILTIN_STYLES:
            with self.subTest(style=style):
                p = build_painting(self.image, style=style, strokes=8,
                                   candidates=4, working_size=48, seed=7)
                self.assertGreater(len(p.operations), 0)
                self.assertEqual(p.width, 48)
                self.assertEqual(p.height, 32)
                canvas = np.empty_like(p.target_rgb, dtype=np.float32)
                canvas[:] = p.base_color
                original_error = ((canvas - p.target_rgb.astype(float)) ** 2).mean()
                for stroke in p.operations:
                    apply_stroke(canvas, stroke)
                final_error = ((canvas - p.target_rgb.astype(float)) ** 2).mean()
                self.assertLess(final_error, original_error)

    def test_deterministic(self):
        p1 = build_painting(self.image, strokes=6, candidates=3, working_size=40, seed=14)
        p2 = build_painting(self.image, strokes=6, candidates=3, working_size=40, seed=14)
        self.assertEqual(p1.operations, p2.operations)


if __name__ == "__main__":
    unittest.main()
