"""Tests the credential-free download path in data/fetch_kaggle.py.

Runs against a throwaway local HTTP server, so no external network is touched.
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import http.server
import importlib.util
import socketserver
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("fetch_kaggle", ROOT / "data" / "fetch_kaggle.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fetch_kaggle"] = module
    spec.loader.exec_module(module)
    return module


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class DirectDownloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

        cls.serve_dir = tempfile.TemporaryDirectory()
        serve_path = Path(cls.serve_dir.name)

        archive = serve_path / "tiny-midi.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("composer/piece.mid", b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60")
            zf.writestr("metadata.csv", "midi_filename,split\ncomposer/piece.mid,train\n")
        cls.checksum = hashlib.sha256(archive.read_bytes()).hexdigest()

        handler = lambda *a, **kw: _QuietHandler(*a, directory=str(serve_path), **kw)
        cls.server = socketserver.TCPServer(("127.0.0.1", 0), handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

        cls.work_dir = tempfile.TemporaryDirectory()
        cls.original_raw_dir = cls.module.RAW_DIR
        cls.module.RAW_DIR = Path(cls.work_dir.name) / "raw"

    @classmethod
    def tearDownClass(cls):
        cls.module.RAW_DIR = cls.original_raw_dir
        cls.server.shutdown()
        cls.server.server_close()
        cls.serve_dir.cleanup()
        cls.work_dir.cleanup()

    def _source(self, key: str, checksum: str) -> dict:
        return {key: {
            "url": f"http://127.0.0.1:{self.port}/tiny-midi.zip",
            "sha256": checksum,
            "size": "tiny",
            "license": "test",
        }}

    def test_download_extract_and_summarise(self):
        key = "tiny"
        original = self.module.DIRECT_SOURCES
        self.module.DIRECT_SOURCES = self._source(key, self.checksum)
        try:
            dest = self.module.fetch_direct(key)
            self.assertTrue((dest / "composer" / "piece.mid").exists(), "MIDI not extracted")
            self.assertTrue((dest / "metadata.csv").exists(), "metadata not extracted")
            self.assertFalse((self.module.RAW_DIR / f"{key}.zip").exists(), "archive not cleaned up")
            self.assertEqual(self.module.fetch_direct(key), dest)
            self.module.summarise(self.module.RAW_DIR)
        finally:
            self.module.DIRECT_SOURCES = original

    def test_checksum_mismatch_is_rejected(self):
        key = "corrupt"
        original = self.module.DIRECT_SOURCES
        self.module.DIRECT_SOURCES = self._source(key, "0" * 64)
        try:
            with self.assertRaises(SystemExit) as ctx:
                self.module.fetch_direct(key)
            self.assertIn("Checksum mismatch", str(ctx.exception))
            self.assertFalse((self.module.RAW_DIR / f"{key}.zip").exists(),
                             "corrupt archive should be deleted")
        finally:
            self.module.DIRECT_SOURCES = original


class RegistryTest(unittest.TestCase):
    def test_registry_entries_are_well_formed(self):
        module = load_module()
        for key, meta in module.KAGGLE_DATASETS.items():
            self.assertIn("/", meta["slug"], f"{key} slug should be owner/dataset")
            self.assertRegex(meta["slug"], r"^[a-z0-9-_]+/[a-z0-9-_]+$", f"{key} slug looks wrong")
        for key, meta in module.DIRECT_SOURCES.items():
            self.assertTrue(meta["url"].startswith("https://"), key)
            self.assertRegex(meta["sha256"], r"^[0-9a-f]{64}$", key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
