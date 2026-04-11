from __future__ import annotations

import asyncio
import importlib.util
import threading
from typing import AsyncIterator

import numpy as np

from voxcpm2_api.audio import to_numpy_audio
from voxcpm2_api.config import Settings
from voxcpm2_api.runtime.base import (
    BackendAvailability,
    StreamChunk,
    SynthesisBackend,
    SynthesisResult,
)
from voxcpm2_api.schemas import SynthesisRequest
from voxcpm2_api.service import PreparedAudioAssets


class VoxCPMBackend(SynthesisBackend):
    name = "voxcpm"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._load_lock = asyncio.Lock()

    def availability(self) -> BackendAvailability:
        if not self._settings.allow_voxcpm:
            return BackendAvailability(False, "Disabled by configuration")
        if importlib.util.find_spec("voxcpm") is None:
            return BackendAvailability(
                False, "Install optional dependency: pip install -e .[voxcpm]"
            )
        return BackendAvailability(True, "Ready")

    def supports_request(self, request: SynthesisRequest) -> bool:
        return True

    async def warmup(self) -> None:
        await self._ensure_model()

    async def synthesize(
        self, request: SynthesisRequest, assets: PreparedAudioAssets
    ) -> SynthesisResult:
        model = await self._ensure_model()

        def _run() -> np.ndarray:
            waveform = model.generate(**self._build_generation_kwargs(request, assets))
            return to_numpy_audio(waveform)

        waveform = await asyncio.to_thread(_run)
        return SynthesisResult(
            backend=self.name,
            device=self._detect_device(model),
            sample_rate=self._detect_sample_rate(model),
            waveform=waveform,
        )

    async def stream(
        self,
        request: SynthesisRequest,
        assets: PreparedAudioAssets,
    ) -> AsyncIterator[StreamChunk]:
        model = await self._ensure_model()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[StreamChunk | Exception | None] = asyncio.Queue()

        def _worker() -> None:
            try:
                for sequence, chunk in enumerate(
                    model.generate_streaming(**self._build_generation_kwargs(request, assets))
                ):
                    payload = StreamChunk(
                        backend=self.name,
                        device=self._detect_device(model),
                        sample_rate=self._detect_sample_rate(model),
                        sequence=sequence,
                        waveform=to_numpy_audio(chunk),
                    )
                    loop.call_soon_threadsafe(queue.put_nowait, payload)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as exc:  # pragma: no cover - surfaced to caller
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def _ensure_model(self):
        if self._model is not None:
            return self._model

        async with self._load_lock:
            if self._model is not None:
                return self._model

            from voxcpm import VoxCPM

            def _load():
                return VoxCPM.from_pretrained(
                    hf_model_id=self._settings.model_source,
                    cache_dir=self._settings.model_cache_dir,
                    local_files_only=self._settings.local_files_only,
                    load_denoiser=self._settings.load_denoiser,
                    optimize=self._settings.optimize_model,
                )

            self._model = await asyncio.to_thread(_load)
            return self._model

    def _build_generation_kwargs(
        self,
        request: SynthesisRequest,
        assets: PreparedAudioAssets,
    ) -> dict[str, object]:
        return {
            "text": request.text,
            "prompt_text": request.prompt_text,
            "prompt_wav_path": assets.prompt_audio_path,
            "reference_wav_path": assets.reference_audio_path,
            "cfg_value": request.cfg_value,
            "inference_timesteps": request.inference_timesteps,
            "min_len": request.min_len,
            "max_len": request.max_len,
            "normalize": request.normalize_text
            if request.normalize_text is not None
            else self._settings.default_normalize_text,
            "denoise": request.denoise_conditioning_audio
            if request.denoise_conditioning_audio is not None
            else self._settings.default_denoise_conditioning_audio,
        }

    def _detect_sample_rate(self, model) -> int:
        tts_model = getattr(model, "tts_model", None)
        return int(getattr(tts_model, "sample_rate", self._settings.sample_rate_fallback))

    def _detect_device(self, model) -> str:
        tts_model = getattr(model, "tts_model", None)
        return str(getattr(tts_model, "device", "cpu"))
