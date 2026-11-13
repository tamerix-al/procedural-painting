"""Offline tests for Streamlit Community Cloud's Primitive bootstrap."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from painting import primitive_backend as backend


class TestPrimitiveCloudSetup(unittest.TestCase):
    def test_existing_binary_does_not_install_again(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / 'primitive'
            binary.write_text('existing')
            with patch.object(backend, 'locate_primitive', return_value=binary), \
                 patch.object(backend.subprocess, 'run') as run:
                self.assertEqual(backend.install_primitive_if_missing(), binary)
                run.assert_not_called()

    def test_build_uses_pinned_dependencies_and_publishes_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'tools' / 'primitive'

            def run(args, **kwargs):
                self.assertEqual(args[1:4], ['build', '-mod=mod', '-o'])
                self.assertEqual(args[-1], 'github.com/fogleman/primitive')
                self.assertEqual(kwargs['env']['GOTOOLCHAIN'], 'local')
                module = (Path(kwargs['cwd']) / 'go.mod').read_text()
                self.assertIn('golang.org/x/image v0.27.0', module)
                self.assertIn('github.com/fogleman/primitive v0.0.0-20200504002142-0373c216458b', module)
                self.assertNotIn('@latest', ' '.join(args))
                self.assertFalse(binary.exists(), 'Do not publish unfinished binaries')
                Path(args[4]).write_bytes(b'fake primitive executable')
                return subprocess.CompletedProcess(args, 0, '', '')

            with patch.object(backend, '__file__', str(root / 'painting' / 'primitive_backend.py')), \
                 patch.object(backend, 'locate_primitive', side_effect=lambda: binary if binary.is_file() else None), \
                 patch.object(backend.shutil, 'which', return_value='/usr/bin/go'), \
                 patch.object(backend.subprocess, 'run', side_effect=run):
                self.assertEqual(backend.install_primitive_if_missing(), binary)
                self.assertTrue(binary.stat().st_mode & 0o111)
                self.assertEqual(binary.read_bytes(), b'fake primitive executable')

    def test_build_failure_is_reported_without_creating_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'tools' / 'primitive'
            with patch.object(backend, '__file__', str(root / 'painting' / 'primitive_backend.py')), \
                 patch.object(backend, 'locate_primitive', return_value=None), \
                 patch.object(backend.shutil, 'which', return_value='/usr/bin/go'), \
                 patch.object(backend.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, '', 'build failed')):
                with self.assertRaisesRegex(RuntimeError, 'build failed'):
                    backend.install_primitive_if_missing()
                self.assertFalse(binary.exists())

    def test_missing_go_reports_cloud_fix(self):
        with patch.object(backend, 'locate_primitive', return_value=None), \
             patch.object(backend.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'packages.txt'):
                backend.install_primitive_if_missing()


if __name__ == '__main__':
    unittest.main()
