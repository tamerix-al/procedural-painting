"""Tests for current GRAIN and image-guided growing TRACE; no legacy TRACE modes in UI."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from painting.engine import apply_stroke, scale_stroke
from painting.graphic import GRAPHIC_STYLES, build_graphic_painting
from painting.layout import LayoutOptions
from painting.project_io import export_project, import_project
from painting.video import export_video


class GraphicTests(unittest.TestCase):
    def setUp(self):
        h, w = 180, 240
        y, x = np.mgrid[:h, :w]
        self.source = np.stack((np.uint8(np.clip(180 + x*.29, 0, 255)),
                                np.uint8(np.clip(220 - y*.55, 0, 255)),
                                np.uint8(95 + 60*np.sin(x/22)**2)), axis=2)

    def test_only_current_trace_is_exposed(self):
        self.assertEqual(set(GRAPHIC_STYLES), {'grain', 'trace'})
        with self.assertRaises(ValueError):
            build_graphic_painting(self.source, 'trace_continuous', marks=200)

    def test_each_style_builds_and_renders(self):
        for style in GRAPHIC_STYLES:
            with self.subTest(style=style):
                model = build_graphic_painting(self.source, style, marks=600,
                                               working_size=260, seed=12)
                self.assertEqual(model.style, style)
                self.assertEqual(len(model.operations), 600)
                canvas = np.empty((model.height, model.width, 3), np.float32)
                canvas[:] = model.base_color
                for mark in model.operations:
                    apply_stroke(canvas, mark)
                self.assertTrue(np.isfinite(canvas).all())
                self.assertGreater(np.max(np.abs(canvas-np.asarray(model.base_color))), 12)
                twice = scale_stroke(model.operations[0], 2., 2.)
                self.assertNotEqual(twice.geometry, model.operations[0].geometry)

    def test_determinism_and_zip_roundtrip(self):
        for style in GRAPHIC_STYLES:
            a = build_graphic_painting(self.source, style, marks=350, working_size=260, seed=7)
            b = build_graphic_painting(self.source, style, marks=350, working_size=260, seed=7)
            self.assertEqual(a.operations, b.operations)
            packed = export_project(self.source, a, LayoutOptions())
            original, restored, _, _ = import_project(packed)
            np.testing.assert_array_equal(original, self.source)
            self.assertEqual(restored.operations, a.operations)

    def test_growing_trace_is_mostly_joined(self):
        model = build_graphic_painting(self.source, 'trace', marks=2500, working_size=260, seed=32)
        self.assertTrue(all(m.kind == 'polyline' and len(m.geometry[0]) >= 3
                            for m in model.operations))
        joined = [a.geometry[0][-1] == b.geometry[0][0]
                  for a, b in zip(model.operations, model.operations[1:])]
        self.assertGreater(sum(joined) / len(joined), .975)
        # Each stroke draws a segment with nonzero length, never a teleporting line.
        self.assertTrue(all(m.geometry[0][0] != m.geometry[0][-1] for m in model.operations))
        for i in range(50):
            stroke = model.operations[i]
            self.assertGreaterEqual(stroke.geometry[0][0][0], 0)
            self.assertLessEqual(stroke.geometry[0][-1][0], model.width)

    def test_trace_starts_locally_and_grows(self):
        model = build_graphic_painting(self.source, 'trace', marks=3000, working_size=260, seed=11)
        starts = np.array([m.geometry[0][0] for m in model.operations[:80]])
        ends = np.array([m.geometry[0][-1] for m in model.operations[:80]])
        # Early drawing remains a connected wandering trail, not a full raster sweep.
        self.assertGreater(np.mean(np.linalg.norm(starts - ends, axis=1)), 1)
        self.assertGreater(np.std(starts[:, 0]), 2)

    def test_trace_improves_image_with_more_segments(self):
        # Improvement is evaluated on a generic synthetic RGB image, no semantic objects.
        model = build_graphic_painting(self.source, 'trace', marks=6000,
                                       working_size=260, seed=44)
        canvas = np.full((model.height, model.width, 3), model.base_color, np.float32)
        initial = float(np.mean((canvas - model.target_rgb)**2))
        for stroke in model.operations:
            apply_stroke(canvas, stroke)
        final = float(np.mean((canvas - model.target_rgb)**2))
        self.assertLess(final, initial*.92)

    def test_graphic_video_with_reveal(self):
        model = build_graphic_painting(self.source, 'trace', marks=500,
                                       working_size=260, seed=4)
        with tempfile.TemporaryDirectory() as temp:
            outfile = Path(temp) / 'trace.mp4'
            export_video(model, self.source, outfile, seconds=1., fps=8,
                         output_size=(360, 640), show_original=True,
                         reveal_mode='soft_wipe', reveal_seconds=.25,
                         original_hold_seconds=.25, pre_reveal_hold_seconds=0.)
            self.assertTrue(outfile.is_file())
            self.assertGreater(outfile.stat().st_size, 1000)
            cap = cv2.VideoCapture(str(outfile))
            self.assertTrue(cap.isOpened())
            cap.release()

    def test_validation(self):
        with self.assertRaises(ValueError):
            build_graphic_painting(self.source, 'trace', marks=12)
        with self.assertRaises(ValueError):
            build_graphic_painting(self.source, 'grain', marks=300, mark_size=7)


if __name__ == '__main__':
    unittest.main()
