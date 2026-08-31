from __future__ import annotations

import asyncio
import types

from voxcpm2_api.config import Settings
from voxcpm2_api.runtime.voxcpm_backend import VoxCPMBackend


def _install_fake_voxcpm(monkeypatch, sentinel):
    fake_module = types.SimpleNamespace(
        VoxCPM=types.SimpleNamespace(from_pretrained=lambda **_: sentinel)
    )
    monkeypatch.setitem(__import__("sys").modules, "voxcpm", fake_module)


def _install_fake_torch(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False),
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)


def test_voxcpm_model_load_is_inline_on_macos(monkeypatch):
    sentinel = object()
    _install_fake_voxcpm(monkeypatch, sentinel)
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr("voxcpm2_api.runtime.voxcpm_backend.platform.system", lambda: "Darwin")

    async def fail_to_thread(*args, **kwargs):
        raise AssertionError("asyncio.to_thread should not be used for macOS model load")

    monkeypatch.setattr(asyncio, "to_thread", fail_to_thread)

    backend = VoxCPMBackend(Settings())

    model = asyncio.run(backend._ensure_model())

    assert model is sentinel


def test_voxcpm_model_load_uses_worker_thread_off_macos(monkeypatch):
    sentinel = object()
    _install_fake_voxcpm(monkeypatch, sentinel)
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr("voxcpm2_api.runtime.voxcpm_backend.platform.system", lambda: "Linux")

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    backend = VoxCPMBackend(Settings())

    model = asyncio.run(backend._ensure_model())

    assert model is sentinel
