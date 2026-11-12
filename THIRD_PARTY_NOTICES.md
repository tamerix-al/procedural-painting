# Third-party notices

## Primitive (Michael Fogleman)

The optional external renderer and optimizer `primitive` is maintained at
https://github.com/fogleman/primitive and licensed under the MIT License.
The executable is **not shipped** with this project. The setup script installs
the upstream application on the user's computer. Any copies redistributed
should include its MIT license text and copyright notice.

Copyright (C) 2016 Michael Fogleman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## anidoodle (Alex Greenshpun)

The new `painting/oil_brush.py` is an adapted, original Python implementation
of the bristle construction ideas and brush algorithm described in
`skills/anidoodle/engine/src/canvas-core/paintedOilKit.ts` from
https://github.com/alexgreensh/anidoodle. We translated and modified the
approach for NumPy/OpenCV (not a verbatim copy or a drop-in use of anidoodle's
TypeScript renderer). The upstream project is licensed under Apache-2.0.

Copyright 2026 Alex Greenshpun.

Changes relative to upstream include: porting from TypeScript canvas APIs to
Python NumPy/OpenCV, ROI-based compositing, simplifying the impasto relief and
linen texture, and bridging to this project's stroke JSON and MP4 renderer.

License text: `THIRD_PARTY_LICENSES/Apache-2.0.txt`.
The original upstream project and documentation remain available at:
https://github.com/alexgreensh/anidoodle.


## GRAIN / TRACE (self-contained graphic modes)

The `painting/graphic.py` and `painting/trace_growth.py` modules were written independently for this project.
It uses general algorithms for image-driven stippling, structure tensors, and
image-guided growing paths and wavy strokes; no source code from anidoodle or from the GPL-3.0 string-art
repository is copied into this module. The references informed the design only.
