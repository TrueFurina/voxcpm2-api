"""Lightweight ASR wrapper around faster-whisper.

Used by the /v1/transcribe endpoint and the Ultimate Cloning flow to
auto-transcribe reference audio so the model can reproduce every vocal nuance.
"""

from __future__ import annotations

import asyncio
import importlib.util
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from voxcpm2_api.audio import decode_base64_audio


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str
    segments: list[TranscriptionSegment] = field(default_factory=list)


def is_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


class ASRService:
    """Lazy-loading Whisper ASR that runs inference in a thread pool."""

    def __init__(self, model_size: str = "base", device: str = "auto"):
        self._model_size = model_size
        self._device = device
        self._model = None
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        compute_type = "int8"
        device = self._device
        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    # faster-whisper doesn't support MPS; fall back to CPU
                    device = "cpu"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

        self._model = WhisperModel(self._model_size, device=device, compute_type=compute_type)
        return self._model

    async def transcribe(
        self,
        audio_base64: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        async with self._get_lock():
            model = self._ensure_model()

        def _run() -> TranscriptionResult:
            # Write audio to a temp file – faster-whisper needs a file path
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(decode_base64_audio(audio_base64))

            try:
                segments_iter, info = model.transcribe(
                    str(tmp_path),
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                )
                segments = []
                texts = []
                for seg in segments_iter:
                    segments.append(
                        TranscriptionSegment(start=seg.start, end=seg.end, text=seg.text.strip())
                    )
                    texts.append(seg.text.strip())

                return TranscriptionResult(
                    text=" ".join(texts),
                    language=info.language,
                    segments=segments,
                )
            finally:
                tmp_path.unlink(missing_ok=True)

        return await asyncio.to_thread(_run)
