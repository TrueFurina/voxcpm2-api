from __future__ import annotations

import base64
import io
import wave
from typing import Any

import numpy as np


def to_numpy_audio(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()

    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("Audio output must contain at least one sample")
    if array.ndim > 1:
        array = array.reshape(-1)
    return np.clip(array, -1.0, 1.0)


def waveform_to_wav_bytes(waveform: Any, sample_rate: int) -> bytes:
    pcm = np.int16(to_numpy_audio(waveform) * 32767)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def waveform_to_base64(waveform: Any, sample_rate: int) -> str:
    return base64.b64encode(waveform_to_wav_bytes(waveform, sample_rate)).decode("ascii")


def decode_base64_audio(b64: str) -> bytes:
    """Decode a base64-encoded audio blob back to raw bytes."""
    try:
        return base64.b64decode(b64, validate=True)
    except Exception as exc:
        raise ValueError("audio_base64 must be valid base64-encoded audio data") from exc


def waveform_chunk_to_base64(waveform: Any, chunk_format: str, sample_rate: int) -> str:
    if chunk_format == "wav":
        payload = waveform_to_wav_bytes(waveform, sample_rate)
    else:
        payload = np.int16(to_numpy_audio(waveform) * 32767).tobytes()
    return base64.b64encode(payload).decode("ascii")
