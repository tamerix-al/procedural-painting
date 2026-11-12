import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from painting.engine import build_painting, apply_stroke, scale_stroke
from painting.layout import LayoutOptions, FrameRenderer, preview_layout
from painting.video import export_video


class TestLayoutAndExport(unittest.TestCase):
    def setUp(self):
        self.image = np.full((80,120,3), [170,120,50], np.uint8)
        self.image[:, :60] = [50,100,160]
        self.opts = LayoutOptions(channel='Мой канал', technique='Треугольники',
                                  painting_title='Очень длинное название картины с несколькими словами',
                                  artist='Имя художника', year='1890')

    def test_preview_and_cyrillic(self):
        preview=preview_layout(self.image,'triangles',self.opts,(540,960))
        self.assertEqual(preview.shape,(960,540,3))
        self.assertGreater(len(np.unique(preview.reshape(-1,3),axis=0)),5)

    def test_rasterizes_at_output_resolution(self):
        painting=build_painting(self.image,strokes=20,candidates=4,working_size=60)
        renderer=FrameRenderer((120,80),(540,960),self.opts,'triangles')
        _,_,w,h=renderer.art_rect
        canvas=np.empty((h,w,3),np.float32);canvas[:]=painting.base_color
        for stroke in painting.operations:
            apply_stroke(canvas,scale_stroke(stroke,w/painting.width,h/painting.height))
        self.assertEqual(canvas.shape,(h,w,3))
        composed=renderer.compose(canvas.astype(np.uint8),len(painting.operations),len(painting.operations))
        self.assertEqual(composed.shape,(960,540,3))

    def test_export_video(self):
        painting=build_painting(self.image,strokes=20,candidates=4,working_size=60)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'render.mp4'
            export_video(painting,self.image,path,seconds=.5,fps=8,
                         output_size=(320,568),layout=self.opts)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size,1000)


if __name__=='__main__':
    unittest.main()
