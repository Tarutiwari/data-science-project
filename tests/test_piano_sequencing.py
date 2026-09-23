"""Tests the split-aware sequencing in piano.py without music21 or TensorFlow.

piano.py imports heavy libraries at module level, so this test installs minimal
stubs in sys.modules and then imports it. That keeps the import honest: if the
script's imports drift, this test fails.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _stub(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def load_piano():
    """Import piano.py with stand-in libraries for the ML/DSP dependencies."""
    if "piano_under_test" in sys.modules:
        return sys.modules["piano_under_test"]

    for sub in ("chord", "converter", "instrument", "note", "stream"):
        _stub(f"music21.{sub}")
    _stub("music21", chord=types.SimpleNamespace(), converter=types.SimpleNamespace(),
          instrument=types.SimpleNamespace(), note=types.SimpleNamespace(),
          stream=types.SimpleNamespace())

    pyplot = _stub("matplotlib.pyplot")
    _stub("matplotlib", use=lambda *_: None, pyplot=pyplot)
    _stub("seaborn", lineplot=lambda *a, **k: types.SimpleNamespace(set=lambda *a, **k: None))

    _stub("tensorflow.random", set_seed=lambda *_: None)
    _stub("tensorflow.keras.callbacks", EarlyStopping=object, ReduceLROnPlateau=object)
    _stub("tensorflow.keras.layers", LSTM=object, Dense=object, Dropout=object, Input=object)
    _stub("tensorflow.keras.models", Sequential=object)
    _stub("tensorflow.keras.optimizers", Adamax=object)
    _stub("tensorflow.keras", callbacks=sys.modules["tensorflow.keras.callbacks"],
          layers=sys.modules["tensorflow.keras.layers"],
          models=sys.modules["tensorflow.keras.models"],
          optimizers=sys.modules["tensorflow.keras.optimizers"])
    _stub("tensorflow", random=sys.modules["tensorflow.random"],
          keras=sys.modules["tensorflow.keras"])

    spec = importlib.util.spec_from_file_location("piano_under_test", ROOT / "piano.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["piano_under_test"] = module
    spec.loader.exec_module(module)
    return module


class SequencingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.piano = load_piano()

    def test_windows_never_cross_piece_boundaries(self):
        """Two pieces with disjoint alphabets: no window may mix them."""
        length = 4
        pieces = {
            "train": [["a"] * 10] + [["b"] * 10],
            "validation": [["c"] * 10],
            "test": [["d"] * 10],
        }
        mapping = {t: i for i, t in enumerate("abcd")}
        sequences = self.piano.sequences_from_pieces(pieces, mapping, length)

        self.assertEqual(sequences["train"][0].shape, (12, length))
        self.assertEqual(sequences["validation"][0].shape, (6, length))
        self.assertEqual(sequences["test"][0].shape, (6, length))

        for split, (X, y) in sequences.items():
            expected = mapping[{"train": "a", "validation": "c", "test": "d"}[split]]
            if split == "train":
                for window, target in zip(X, y):
                    self.assertEqual(len(set(window)), 1, f"window mixes pieces: {window}")
                    self.assertEqual(target, window[0], "target jumped to another piece")
            else:
                self.assertTrue((y == expected).all(), f"{split} leaked another piece")

    def test_concatenating_pieces_would_be_detectable(self):
        """Guard against a regression to the old flat-corpus behaviour."""
        length = 3
        pieces = {"train": [["a"] * 5, ["b"] * 5]}
        mapping = {"a": 0, "b": 1}
        X, y = self.piano.sequences_from_pieces(pieces, mapping, length)["train"]
        self.assertEqual(len(X), 4)
        mixed = [w for w in X if len(set(w)) > 1]
        self.assertEqual(mixed, [], "a boundary window was generated")

    def test_rare_filtering_drops_tokens_and_short_pieces(self):
        pieces = {
            "train": [["a", "a", "a", "b"], ["a", "a", "a", "a", "a", "a"]],
            "test": [["rare", "a", "a", "a", "a"]],
        }
        filtered, counts = self.piano.filter_rare_pieces(pieces, min_count=2, min_len=3)
        self.assertNotIn("rare", counts)
        self.assertEqual(counts, {"a": 13})
        self.assertEqual(filtered["train"][0], ["a", "a", "a"])
        self.assertEqual(filtered["test"][0], ["a", "a", "a", "a"])
        self.assertEqual(sum(map(len, filtered["train"])), 9)

    def test_longest_piece_keeps_every_window(self):
        pieces = {"train": [["x"] * 10]}
        X, y = self.piano.sequences_from_pieces(pieces, {"x": 0}, length=2)["train"]
        self.assertEqual(len(y), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
