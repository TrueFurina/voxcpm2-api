from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class HardwareProfile:
    operating_system: str
    architecture: str
    python_version: str
    apple_silicon: bool
    cuda_available: bool
    cuda_device_count: int
    mps_available: bool
    recommended_backend: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _detect_torch():
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def detect_hardware(torch_module=None) -> HardwareProfile:
    torch_module = torch_module if torch_module is not None else _detect_torch()
    operating_system = platform.system().lower()
    architecture = platform.machine().lower()
    apple_silicon = operating_system == "darwin" and architecture in {"arm64", "aarch64"}

    cuda_available = False
    cuda_device_count = 0
    mps_available = False
    if torch_module is not None:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None:
            cuda_available = bool(cuda.is_available())
            if cuda_available:
                cuda_device_count = int(cuda.device_count())

        backends = getattr(torch_module, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        if mps is not None:
            mps_available = bool(mps.is_available())

    if operating_system == "linux" and cuda_available:
        recommended_backend = "nanovllm"
    elif cuda_available or mps_available or apple_silicon:
        recommended_backend = "voxcpm"
    else:
        recommended_backend = "voxcpm"

    return HardwareProfile(
        operating_system=operating_system,
        architecture=architecture,
        python_version=sys.version.split()[0],
        apple_silicon=apple_silicon,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        mps_available=mps_available,
        recommended_backend=recommended_backend,
    )
