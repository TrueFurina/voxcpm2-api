import base64

import numpy as np

from voxcpm2_api.audio import waveform_chunk_to_base64, waveform_to_wav_bytes


def test_waveform_to_wav_bytes_has_riff_header() -> None:
    payload = waveform_to_wav_bytes(np.array([0.0, 0.5, -0.5], dtype=np.float32), 24000)
    assert payload[:4] == b"RIFF"


def test_waveform_chunk_pcm16_roundtrip() -> None:
    encoded = waveform_chunk_to_base64(np.array([0.25], dtype=np.float32), "pcm16", 24000)
    raw = base64.b64decode(encoded)
    assert len(raw) == 2
