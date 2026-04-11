from types import SimpleNamespace

from voxcpm2_api.hardware import detect_hardware


def test_detect_hardware_prefers_nanovllm_for_linux_cuda(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")

    profile = detect_hardware(fake_torch)

    assert profile.cuda_available is True
    assert profile.recommended_backend == "nanovllm"
