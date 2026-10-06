"""TrOCR decoding: no_repeat_ngram_size defaults to 0 and reaches generate().

Checkpoints from optimized_training.py store no_repeat_ngram_size=3 in their
generation_config; unless the call overrides it, generate() inherits it and forbids
legitimate repetitions inside a line.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engines.trocr_engine import TrOCREngine


class _FakeInference:
    def __init__(self):
        self.calls = []

    def transcribe_line(self, image, **kwargs):
        self.calls.append(kwargs)
        return "text", 0.9, [0.9]


def _engine(**state):
    engine = TrOCREngine()
    engine.model = _FakeInference()
    for k, v in state.items():
        setattr(engine, k, v)
    return engine


def test_default_turns_the_inherited_ngram_block_off():
    engine = _engine()
    engine.transcribe_line(np.zeros((32, 64, 3), dtype=np.uint8), config={})
    assert engine.model.calls[0]["no_repeat_ngram_size"] == 0


def test_value_from_load_config_is_used():
    engine = _engine(_no_repeat_ngram_size=3)
    engine.transcribe_line(np.zeros((32, 64, 3), dtype=np.uint8), config={})
    assert engine.model.calls[0]["no_repeat_ngram_size"] == 3


def test_transcribe_line_signature_defaults_to_zero():
    import inspect
    from inference_page import TrOCRInference
    sig = inspect.signature(TrOCRInference.transcribe_line)
    assert sig.parameters["no_repeat_ngram_size"].default == 0


def test_batch_cli_flag_defaults_to_zero(monkeypatch, tmp_path):
    import batch_processing
    monkeypatch.setattr(sys, "argv", ["batch_processing.py", "--input-folder", str(tmp_path),
                                      "--engine", "TrOCR", "--model-id", "x/y"])
    args = batch_processing.parse_args()
    assert args.no_repeat_ngram_size == 0
