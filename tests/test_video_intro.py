"""Video intro is optional, artwork-only, and non-destructive."""
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from painting.engine import build_painting
from painting.video import INTRO_MODES, export_video, intro_transition_frame


class IntroVideoTests(unittest.TestCase):
    def setUp(self):
        self.original = np.full((24, 360, 3), [180, 55, 40], dtype=np.uint8)
        self.blank = np.full((24, 360, 3), [245, 242, 235], dtype=np.uint8)

    def test_both_modes_have_exact_endpoints(self):
        for mode in INTRO_MODES:
            with self.subTest(mode=mode):
                np.testing.assert_array_equal(intro_transition_frame(self.original, self.blank, mode, 0), self.original)
                np.testing.assert_array_equal(intro_transition_frame(self.original, self.blank, mode, 1), self.blank)
                mid = intro_transition_frame(self.original, self.blank, mode, .5)
                self.assertFalse(np.array_equal(mid, self.original))
                self.assertFalse(np.array_equal(mid, self.blank))

    def test_soft_wipe_erases_from_left_to_right(self):
        mid = intro_transition_frame(self.original, self.blank, 'soft_wipe_out', .5)
        np.testing.assert_array_equal(mid[12, 10], self.blank[12, 10])
        np.testing.assert_array_equal(mid[12, -10], self.original[12, -10])
        pixels = mid[12, :, 0]
        self.assertGreater(np.count_nonzero((pixels > 180) & (pixels < 245)), 10)

    def test_fade_out_changes_entire_picture(self):
        mid = intro_transition_frame(self.original, self.blank, 'fade_out', .5)
        np.testing.assert_array_equal(mid[5, 5], mid[-5, -5])
        self.assertGreater(mid[5, 5, 0], 180)
        self.assertLess(mid[5, 5, 0], 245)

    def test_invalid_mode_and_shape(self):
        with self.assertRaises(ValueError):
            intro_transition_frame(self.original, self.blank, 'unknown', .5)
        with self.assertRaises(ValueError):
            intro_transition_frame(self.original, self.blank[:1], 'fade_out', .5)

    def test_video_intro_frame_count_and_no_recalculation(self):
        original = np.empty((40, 60, 3), dtype=np.uint8)
        original[:, :] = [35, 130, 180]
        original[:, 30:] = [185, 70, 50]
        p = build_painting(original, style='triangles', strokes=12,
                           candidates=3, working_size=60, seed=10)
        saved_ops = list(p.operations)
        for mode in INTRO_MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / 'intro.mp4'
                progress = []
                export_video(
                    p, original, dest, seconds=.5, fps=8, output_size=(320, 568),
                    show_original=True, reveal_seconds=.5, pre_reveal_hold_seconds=.25,
                    original_hold_seconds=.25, show_intro=True,
                    intro_hold_seconds=.5, intro_transition_seconds=.5,
                    intro_mode=mode, progress=lambda done, total: progress.append((done, total)),
                )
                with imageio.get_reader(str(dest), format='FFMPEG') as reader:
                    # 4 hold + 4 intro transition + 4 paint + 2 pause + 4 reveal + 2 final.
                    self.assertEqual(reader.count_frames(), 20)
                self.assertEqual(progress[-1], (20, 20))
                self.assertGreater(dest.stat().st_size, 3000)
                self.assertEqual(p.operations, saved_ops)

    def test_disable_intro_preserves_old_video_timing(self):
        original = np.full((40, 60, 3), 180, dtype=np.uint8)
        p = build_painting(original, style='triangles', strokes=10,
                           candidates=3, working_size=60, seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'no_intro.mp4'
            export_video(p, original, dest, seconds=.5, fps=8,
                         output_size=(320, 568), show_intro=False,
                         show_original=False)
            with imageio.get_reader(str(dest), format='FFMPEG') as reader:
                self.assertEqual(reader.count_frames(), 4)

    def test_negative_intro_durations_fail_fast(self):
        original = np.full((40, 60, 3), 180, dtype=np.uint8)
        p = build_painting(original, style='triangles', strokes=10,
                           candidates=3, working_size=60, seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            for kwargs in ({'intro_hold_seconds': -1}, {'intro_transition_seconds': -1}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    export_video(p, original, Path(tmp) / 'bad.mp4', **kwargs)


if __name__ == '__main__':
    unittest.main()
