import tempfile
import unittest
from pathlib import Path

import numpy as np
from painting.engine import apply_stroke, scale_stroke
from painting.layout import LayoutOptions
from painting.primitive_backend import parse_primitive_svg, run_primitive
from painting.project_io import export_project, import_project
from painting.video import export_video


SAMPLE_SVG='''<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="128" height="96">
<rect width="128" height="96" fill="#777777" />
<g transform="scale(2.000000) translate(0.5 0.5)">
<polygon fill="#00a020" fill-opacity="0.500000" points="0,0 40,0 0,40" />
<ellipse fill="#eabf08" fill-opacity="0.750000" cx="32" cy="22" rx="9" ry="5" />
<g transform="translate(20 30) rotate(15) scale(12 6)"><rect fill="#a02080" fill-opacity="0.6" x="-0.5" y="-0.5" width="1" height="1" /></g>
<g transform="translate(25.0 20.0) rotate(24.0) scale(8.0 4.0)"><ellipse fill="#99ffff" fill-opacity="0.3" cx="0" cy="0" rx="1" ry="1" /></g>
<path stroke="#111111" stroke-opacity="0.5" fill="none" d="M 3 5 Q 12 38, 35 12" stroke-width="2" />
</g></svg>'''


class TestPrimitiveBridge(unittest.TestCase):
    def test_svg_geometry_and_order(self):
        p=parse_primitive_svg(SAMPLE_SVG,'mix')
        self.assertEqual((p.width,p.height),(128,96))
        self.assertEqual(p.base_color,(119,119,119))
        self.assertEqual([s.kind for s in p.operations],['poly','ellipse','poly','ellipse','polyline'])
        self.assertAlmostEqual(p.operations[0].geometry[0][0],1)
        self.assertAlmostEqual(p.operations[0].geometry[0][1],1)
        self.assertAlmostEqual(p.operations[0].alpha,0.5)
        arr=np.empty((96,128,3),np.float32)
        arr[:]=p.base_color
        for op in p.operations:
            apply_stroke(arr,op)
        self.assertGreater(arr.max()-arr.min(),50)

    def test_project_save_load_and_video(self):
        source=np.zeros((24,32,3),np.uint8)
        source[:]=[150,100,60]
        p=parse_primitive_svg(SAMPLE_SVG,'mix')
        opts=LayoutOptions(channel='Канал тест',painting_title='Тест')
        raw=export_project(source,p,opts,SAMPLE_SVG.encode())
        source2,p2,layout,svg=import_project(raw)
        self.assertTrue(np.array_equal(source,source2))
        self.assertEqual(p2.operations,p.operations)
        self.assertEqual(layout['channel'],'Канал тест')
        self.assertTrue(svg.startswith(b'<svg'))
        with tempfile.TemporaryDirectory() as t:
            dest=Path(t)/'out.mp4'
            export_video(p2,source2,dest,seconds=0.5,fps=8,
                         output_size=(320,568),layout=opts)
            self.assertGreater(dest.stat().st_size,1000)

    def test_primitive_cli_adapter(self):
        # Fake the CLI process, so the integration contract can be tested on
        # Linux without shipping upstream Go binaries or requiring internet.
        with tempfile.TemporaryDirectory() as t:
            binary=Path(t)/'primitive'
            binary.write_text('#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\nargs=sys.argv\np=Path(args[args.index("-o")+1])\np.write_text('+repr(SAMPLE_SVG)+',encoding="utf8")\nprint("1: t=0.003, score=0.5")\nprint("2: t=0.005, score=0.4")\n',encoding='utf-8')
            binary.chmod(0o755)
            source=np.full((24,32,3),170,np.uint8)
            calls=[]
            p,raw=run_primitive(source,'triangles',count=2,analysis_size=256,
                                executable=str(binary),progress=lambda i,n:calls.append((i,n)))
            self.assertEqual(len(p.operations),5)
            self.assertEqual(calls,[(1,2),(2,2)])
            self.assertIn(b'<svg',raw)


if __name__=='__main__':
    unittest.main()
