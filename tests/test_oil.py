"""Oil engine regression, including re-opening v0.4 projects and rendering MP4."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from painting.engine import build_painting, apply_stroke, scale_stroke
from painting.oil_brush import paint_oil, validate_oil_geometry
from painting.layout import LayoutOptions
from painting.project_io import export_project, import_project
from painting.video import export_video


class TestOilRenderer(unittest.TestCase):
    def setUp(self):
        img=np.empty((50,80,3),np.uint8)
        img[:,:40]=[195,110,75]
        img[:,40:]=[35,125,170]
        self.image=img

    def test_oil_modes_are_reproducible(self):
        for style in ('oil','dry_brush'):
            with self.subTest(style=style):
                a=build_painting(self.image,style=style,strokes=14,
                                 candidates=5,working_size=70,seed=31)
                b=build_painting(self.image,style=style,strokes=14,
                                 candidates=5,working_size=70,seed=31)
                self.assertGreater(len(a.operations),0)
                self.assertEqual(a.operations,b.operations)
                self.assertTrue(all(op.kind=='oil' for op in a.operations))
                for op in a.operations:validate_oil_geometry(op.geometry)

    def test_compact_strokes_and_size_controls(self):
        compact=build_painting(self.image, style='oil', strokes=48,
                               candidates=6, working_size=70, seed=13,
                               oil_length_ratio=1.6, oil_size=.85)
        ratios=[]
        for stroke in compact.operations:
            pts,width,*_=stroke.geometry
            a=np.asarray(pts,dtype=float)
            ratios.append(float(np.linalg.norm(a[-1]-a[0]))/float(width))
        self.assertLess(float(np.median(ratios)),2.0)
        self.assertLess(float(np.percentile(ratios,90)),2.5)
        elongated=build_painting(self.image, style='oil', strokes=48,
                                 candidates=6, working_size=70, seed=13,
                                 oil_length_ratio=4.0, oil_size=.85)
        ratios_long=[]
        for stroke in elongated.operations:
            pts,width,*_=stroke.geometry
            a=np.asarray(pts,dtype=float)
            ratios_long.append(float(np.linalg.norm(a[-1]-a[0]))/float(width))
        self.assertGreater(float(np.median(ratios_long)),float(np.median(ratios)))
        smaller=build_painting(self.image, style='oil', strokes=48,
                               candidates=6, working_size=70, seed=13,
                               oil_length_ratio=1.6, oil_size=.5)
        self.assertNotEqual(compact.operations,smaller.operations)

    def test_partial_stroke_is_deterministic(self):
        geometry=[[[8.,13.],[24.,20.],[44.,18.]],9.,.28,.5,17]
        a=np.full((46,60,3),128.,np.float32)
        b=a.copy();c=a.copy()
        paint_oil(a,geometry,(200,60,25),progress=.5)
        paint_oil(b,geometry,(200,60,25),progress=.5)
        paint_oil(c,geometry,(200,60,25),progress=1.)
        self.assertTrue(np.array_equal(a,b))
        self.assertFalse(np.array_equal(a,c))

    def test_project_round_trip_and_mp4(self):
        p=build_painting(self.image,style='oil',strokes=24,candidates=6,
                         working_size=70,seed=7)
        layout=LayoutOptions(channel='Живопись',painting_title='Тест масла')
        archive=export_project(self.image,p,layout)
        original,restored,settings,svg=import_project(archive)
        self.assertTrue(np.array_equal(original,self.image))
        self.assertEqual(restored.operations,p.operations)
        self.assertEqual(restored.style,'oil')
        self.assertEqual(settings['channel'],'Живопись')
        self.assertIsNone(svg)
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'oil.mp4'
            export_video(restored,original,target,seconds=.7,fps=8,
                         output_size=(320,568),layout=layout)
            self.assertGreater(target.stat().st_size,5000)

    def test_oil_full_resolution(self):
        p=build_painting(self.image,style='dry_brush',strokes=10,
                         candidates=4,working_size=70,seed=5)
        art=np.empty((100,160,3),np.float32);art[:]=p.base_color
        for op in p.operations:
            apply_stroke(art,scale_stroke(op,160/p.width,100/p.height))
        self.assertGreater(float(art.std()),6.0)

if __name__=='__main__':
    unittest.main()
