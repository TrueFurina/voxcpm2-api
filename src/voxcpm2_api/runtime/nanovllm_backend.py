from __future__ import annotations

import asyncio
import importlib.util
import platform
from typing import AsyncIterator

import numpy as np

from voxcpm2_api.audio import to_numpy_audio
from voxcpm2_api.config import Settings
from voxcpm2_api.hardware import HardwareProfile
from voxcpm2_api.runtime.base import (
    BackendAvailability,
    StreamChunk,
    SynthesisBackend,
    SynthesisResult,
)
from voxcpm2_api.schemas import SynthesisRequest
from voxcpm2_api.service import PreparedAudioAssets


class NanoVLLMBackend(SynthesisBackend):
    name = "nanovllm"

    def __init__(self, settings: Settings, hardware: HardwareProfile):
        self._settings = settings
        self._hardware = hardware
        self._server = None
        self._load_lock = asyncio.Lock()

    def availability(self) -> BackendAvailability:
        if not self._settings.allow_nanovllm:
            return BackendAvailability(False, "Disabled by configuration")
        if platform.system().lower() != "linux":
            return BackendAvailability(False, "Nano-vLLM is Linux-only")
        if not self._hardware.cuda_available:
            return BackendAvailability(False, "Nano-vLLM requires NVIDIA CUDA")
        if importlib.util.find_spec("nanovllm_voxcpm") is None:
            return BackendAvailability(
                False, "Install optional dependency: pip install -e .[nanovllm]"
            )
        return BackendAvailability(True, "Ready")

    def supports_request(self, request: SynthesisRequest) -> bool:
        return not request.has_conditioning

    async def warmup(self) -> None:
        await self._ensure_server()

    async def synthesize(
        self, request: SynthesisRequest, assets: PreparedAudioAssets
    ) -> SynthesisResult:
        del assets
        server = await self._ensure_server()
        chunks = []
        async for chunk in server.generate(target_text=request.text):
            chunks.append(to_numpy_audio(chunk))

        waveform = np.concatenate(chunks, axis=0) if chunks else np.zeros(0, dtype=np.float32)
        return SynthesisResult(
            backend=self.name,
            device=f"cuda:{self._settings.nanovllm_devices[0]}",
            sample_rate=int(getattr(server, "sample_rate", self._settings.sample_rate_fallback)),
            waveform=waveform,
        )

    async def stream(
        self,
        request: SynthesisRequest,
        assets: PreparedAudioAssets,
    ) -> AsyncIterator[StreamChunk]:
        del assets
        server = await self._ensure_server()
        sequence = 0
        async for chunk in server.generate(target_text=request.text):
            yield StreamChunk(
                backend=self.name,
                device=f"cuda:{self._settings.nanovllm_devices[0]}",
                sample_rate=int(
                    getattr(server, "sample_rate", self._settings.sample_rate_fallback)
                ),
                sequence=sequence,
                waveform=to_numpy_audio(chunk),
            )
            sequence += 1

    async def close(self) -> None:
        if self._server is not None:
            maybe_coroutine = self._server.stop()
            if asyncio.iscoroutine(maybe_coroutine):
                await maybe_coroutine
            self._server = None

    async def _ensure_server(self):
        if self._server is not None:
            return self._server

        async with self._load_lock:
            if self._server is not None:
                return self._server

            from nanovllm_voxcpm import VoxCPM

            server = VoxCPM.from_pretrained(
                model=self._settings.model_source,
                devices=self._settings.nanovllm_devices,
                inference_timesteps=10,
                max_num_batched_tokens=self._settings.nanovllm_max_num_batched_tokens,
                max_num_seqs=self._settings.nanovllm_max_num_seqs,
                max_model_len=self._settings.nanovllm_max_model_len,
                gpu_memory_utilization=self._settings.nanovllm_gpu_memory_utilization,
            )
            wait_for_ready = getattr(server, "wait_for_ready", None)
            if wait_for_ready is not None:
                await wait_for_ready()
            self._server = server
            return self._server
