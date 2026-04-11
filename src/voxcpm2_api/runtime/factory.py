from __future__ import annotations

import os
from collections.abc import AsyncIterator

from voxcpm2_api.compat import apply_torch_compat_patches
from voxcpm2_api.config import Settings
from voxcpm2_api.hardware import detect_hardware
from voxcpm2_api.runtime.base import SynthesisBackend, StreamChunk, SynthesisResult
from voxcpm2_api.runtime.nanovllm_backend import NanoVLLMBackend
from voxcpm2_api.runtime.voxcpm_backend import VoxCPMBackend
from voxcpm2_api.schemas import RuntimeResponse, SynthesisRequest
from voxcpm2_api.service import PreparedAudioAssets


class RuntimeOrchestrator:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._hardware = detect_hardware()
        self._backends: dict[str, SynthesisBackend] = {
            "voxcpm": VoxCPMBackend(settings),
            "nanovllm": NanoVLLMBackend(settings, self._hardware),
        }

    async def warmup(self) -> None:
        backend = self._select_backend(SynthesisRequest(text="Warmup", response_format="wav"))
        await backend.warmup()

    async def close(self) -> None:
        for backend in self._backends.values():
            await backend.close()

    def runtime_snapshot(self, request: SynthesisRequest | None = None) -> RuntimeResponse:
        probe_request = request or SynthesisRequest(text="status", response_format="wav")
        try:
            selected_backend_name = self._select_backend(probe_request).name
        except RuntimeError:
            selected_backend_name = "unavailable"
        status = {
            name: {
                "available": backend.availability().available,
                "reason": backend.availability().reason,
            }
            for name, backend in self._backends.items()
        }
        return RuntimeResponse(
            selected_backend=selected_backend_name,
            requested_backend=self._settings.prefer_backend,
            model_source=self._settings.model_source,
            hardware=self._hardware.to_dict(),
            backend_status=status,
        )

    async def synthesize(
        self,
        request: SynthesisRequest,
        assets: PreparedAudioAssets,
    ) -> SynthesisResult:
        backend = self._select_backend(request)
        return await backend.synthesize(request, assets)

    async def stream(
        self,
        request: SynthesisRequest,
        assets: PreparedAudioAssets,
    ) -> AsyncIterator[StreamChunk]:
        backend = self._select_backend(request)
        async for chunk in backend.stream(request, assets):
            yield chunk

    def _select_backend(self, request: SynthesisRequest) -> SynthesisBackend:
        requested = self._settings.prefer_backend
        candidates: list[str]

        if request.has_conditioning:
            candidates = ["voxcpm"]
        elif requested == "auto":
            candidates = [self._hardware.recommended_backend, "voxcpm", "nanovllm"]
        else:
            candidates = [requested, "voxcpm", "nanovllm"]

        seen: set[str] = set()
        for name in candidates:
            if name in seen or name not in self._backends:
                continue
            seen.add(name)
            backend = self._backends[name]
            availability = backend.availability()
            if availability.available and backend.supports_request(request):
                return backend

        detail = "; ".join(
            f"{name}: {backend.availability().reason}" for name, backend in self._backends.items()
        )
        raise RuntimeError(f"No compatible backend is available. {detail}")


def configure_environment(settings: Settings) -> None:
    apply_torch_compat_patches()
    if settings.hf_endpoint:
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint
