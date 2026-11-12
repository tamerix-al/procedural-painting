"""Regression tests for non-destructive stroke timing and artwork-only reveals."""
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from painting.engine import build_painting
from painting.layout import FrameRenderer, LayoutOptions
from painting.video import export_video, stroke_progress, transition_frame, REVEAL_MODES


class VideoTransitionsTests(unittest.TestCase):
    def setUp(self):
        self.painted=np.full((32,48,3),[180,45,25],np.uint8)
        self.original=np.full((32,48,3),[35,160,190],np.uint8)

    def test_moderate_progress_is_smooth_monotonic_and_accelerates_end(self):
        points=[stroke_progress(i/100,'moderate') for i in range(101)]
        self.assertTrue(all(b>=a for a,b in zip(points,points[1:])))
        self.assertAlmostEqual(points[0],0)
        self.assertAlmostEqual(points[-1],1)
        self.assertAlmostEqual(points[90],.8,delta=.01)
        self.assertAlmostEqual(stroke_progress(.3,'uniform'),.3)
        self.assertLess(stroke_progress(.7,'moderate'),.7)
        with self.assertRaises(ValueError):stroke_progress(.5,'magic')

    def test_reveal_modes_have_exact_endpoints(self):
        for mode in REVEAL_MODES:
            with self.subTest(mode=mode):
                np.testing.assert_array_equal(transition_frame(self.painted,self.original,mode,0),self.painted)
                np.testing.assert_array_equal(transition_frame(self.painted,self.original,mode,1),self.original)
                partial=transition_frame(self.painted,self.original,mode,.5)
                self.assertEqual(partial.shape,self.painted.shape)
                self.assertFalse(np.array_equal(partial,self.painted))
                self.assertFalse(np.array_equal(partial,self.original))

    def test_soft_wipe_feather_and_split_separator(self):
        # A wide canvas gives enough room for the actual feather zone.
        painted=np.full((40,400,3),10,np.uint8)
        original=np.full((40,400,3),200,np.uint8)
        soft=transition_frame(painted,original,'soft_wipe',.5)
        self.assertEqual(int(soft[20,0,0]),200)
        self.assertEqual(int(soft[20,-1,0]),10)
        between=soft[20,:,0]
        self.assertGreater(np.count_nonzero((between>10)&(between<200)),10)
        split=transition_frame(painted,original,'split_reveal',.5)
        self.assertEqual(int(split[20,20,0]),200)
        self.assertEqual(int(split[20,-20,0]),10)
        self.assertTrue(np.any(split[20,197:204] != 10))
        fade=transition_frame(painted,original,'crossfade',.5)
        self.assertTrue(np.all(fade==105))

    def test_reveal_applies_only_inside_artwork(self):
        opts=LayoutOptions(channel='Тест',painting_title='Горы')
        renderer=FrameRenderer((48,32),(320,568),opts,'oil')
        a=renderer.compose(transition_frame(self.painted,self.original,'soft_wipe',.5))
        b=renderer.compose(transition_frame(self.painted,self.original,'crossfade',.5))
        x,y,w,h=renderer.art_rect
        # The output frame decorations don't depend on the reveal.
        np.testing.assert_array_equal(a[:y,:,:],b[:y,:,:])
        np.testing.assert_array_equal(a[y+h:,:,:],b[y+h:,:,:])
        self.assertFalse(np.array_equal(a[y:y+h,x:x+w],b[y:y+h,x:x+w]))

    def test_export_and_frame_counts_for_each_mode(self):
        original=np.full((40,60,3),[40,125,190],np.uint8)
        original[:,:30]=[185,115,50]
        p=build_painting(original,style='triangles',strokes=12,
                         candidates=3,working_size=60,seed=12)
        original_ops=list(p.operations)
        for mode in REVEAL_MODES:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                dest=Path(tmp)/'result.mp4'
                export_video(p,original,dest,seconds=.5,fps=8,
                             output_size=(320,568),reveal_seconds=.5,
                             pre_reveal_hold_seconds=.25,
                             original_hold_seconds=.25,
                             reveal_mode=mode,stroke_timing='moderate')
                with imageio.get_reader(str(dest),format='FFMPEG') as reader:
                    self.assertEqual(reader.count_frames(),4+2+4+2)
                self.assertEqual(p.operations,original_ops)
                self.assertGreater(dest.stat().st_size,5000)
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'no_original.mp4'
            export_video(p,original,dest,seconds=.5,fps=8,output_size=(320,568),
                         show_original=False,stroke_timing='uniform')
            with imageio.get_reader(str(dest),format='FFMPEG') as reader:
                self.assertEqual(reader.count_frames(),4)


if __name__=='__main__':
    unittest.main()
