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

    def test_go_install_creates_binary_in_tools(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'tools' / 'primitive'
            def run(args, **kwargs):
                if args[1] == 'version':
                    return subprocess.CompletedProcess(args, 0, 'go version go1.23.2 linux/amd64', '')
                self.assertEqual(args[1:3], ['install', 'github.com/fogleman/primitive@latest'])
                self.assertEqual(kwargs['env']['GOBIN'], str(binary.parent))
                binary.write_bytes(b'fake binary')
                return subprocess.CompletedProcess(args, 0, '', '')
            with patch.object(backend, '__file__', str(root / 'painting' / 'primitive_backend.py')), \
                 patch.object(backend, 'locate_primitive', side_effect=lambda: binary if binary.is_file() else None), \
                 patch.object(backend.shutil, 'which', return_value='/usr/bin/go'), \
                 patch.object(backend.subprocess, 'run', side_effect=run):
                self.assertEqual(backend.install_primitive_if_missing(), binary)
                self.assertTrue(binary.stat().st_mode & 0o111)

    def test_debian_legacy_go_uses_go_get(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'tools' / 'primitive'
            def run(args, **kwargs):
                if args[1] == 'version':
                    return subprocess.CompletedProcess(args, 0, 'go version go1.15.15 linux/amd64', '')
                self.assertEqual(args[1:3], ['get', '-u'])
                self.assertEqual(kwargs['env']['GO111MODULE'], 'off')
                binary.write_bytes(b'fake binary')
                return subprocess.CompletedProcess(args, 0, '', '')
            with patch.object(backend, '__file__', str(root / 'painting' / 'primitive_backend.py')), \
                 patch.object(backend, 'locate_primitive', side_effect=lambda: binary if binary.is_file() else None), \
                 patch.object(backend.shutil, 'which', return_value='/usr/bin/go'), \
                 patch.object(backend.subprocess, 'run', side_effect=run):
                self.assertEqual(backend.install_primitive_if_missing(), binary)

    def test_missing_go_reports_cloud_fix(self):
        with patch.object(backend, 'locate_primitive', return_value=None), \
             patch.object(backend.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'packages.txt'):
                backend.install_primitive_if_missing()


if __name__ == '__main__':
    unittest.main()
