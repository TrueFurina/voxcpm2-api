from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np

from voxcpm2_api.schemas import SynthesisRequest
from voxcpm2_api.service import PreparedAudioAssets


@dataclass(slots=True)
class BackendAvailability:
    available: bool
    reason: str


@dataclass(slots=True)
class SynthesisResult:
    backend: str
    device: str
    sample_rate: int
    waveform: np.ndarray


@dataclass(slots=True)
class StreamChunk:
    backend: str
    device: str
    sample_rate: int
    sequence: int
    waveform: np.ndarray


class SynthesisBackend(ABC):
    name: str

    @abstractmethod
    def availability(self) -> BackendAvailability:
        raise NotImplementedError

    @abstractmethod
    def supports_request(self, request: SynthesisRequest) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def synthesize(
        self,
        request: SynthesisRequest,
        assets: PreparedAudioAssets,
    ) -> SynthesisResult:
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        request: SynthesisRequest,
        assets: PreparedAudioAssets,
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None
