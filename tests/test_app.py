import numpy as np
from fastapi.testclient import TestClient

from voxcpm2_api.config import Settings
from voxcpm2_api.main import create_app
from voxcpm2_api.runtime.base import StreamChunk, SynthesisResult
from voxcpm2_api.schemas import RuntimeResponse, SynthesisRequest


class FakeRuntime:
    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def runtime_snapshot(self, request: SynthesisRequest | None = None) -> RuntimeResponse:
        del request
        return RuntimeResponse(
            selected_backend="voxcpm",
            requested_backend="auto",
            model_source="openbmb/VoxCPM2",
            hardware={"operating_system": "darwin"},
            backend_status={"voxcpm": {"available": True, "reason": "Ready"}},
        )

    async def synthesize(self, request: SynthesisRequest, assets) -> SynthesisResult:
        del request, assets
        return SynthesisResult(
            backend="voxcpm",
            device="cpu",
            sample_rate=24000,
            waveform=np.array([0.0, 0.25, -0.25], dtype=np.float32),
        )

    async def stream(self, request: SynthesisRequest, assets):
        del request, assets
        yield StreamChunk(
            backend="voxcpm",
            device="cpu",
            sample_rate=24000,
            sequence=0,
            waveform=np.array([0.0, 0.25], dtype=np.float32),
        )


def test_json_synthesis_response() -> None:
    app = create_app(settings=Settings(), orchestrator=FakeRuntime())
    with TestClient(app) as client:
        response = client.post(
            "/v1/speech",
            json={"text": "hello", "response_format": "base64"},
        )

    assert response.status_code == 200
    assert response.json()["backend"] == "voxcpm"


def test_health_alias_status_endpoint() -> None:
    app = create_app(settings=Settings(), orchestrator=FakeRuntime())
    with TestClient(app) as client:
        response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_invalid_prompt_audio_returns_422() -> None:
    app = create_app(settings=Settings(), orchestrator=FakeRuntime())
    with TestClient(app) as client:
        response = client.post(
            "/v1/speech",
            json={
                "text": "hello",
                "prompt_text": "context",
                "prompt_audio_base64": "not-valid-base64",
            },
        )

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_websocket_stream() -> None:
    app = create_app(settings=Settings(), orchestrator=FakeRuntime())
    with TestClient(app) as client:
        with client.websocket_connect("/v1/stream") as websocket:
            websocket.send_json({"text": "hello", "chunk_format": "pcm16"})
            started = websocket.receive_json()
            chunk = websocket.receive_json()
            completed = websocket.receive_json()

    assert started["type"] == "session.started"
    assert chunk["type"] == "audio.chunk"
    assert completed["type"] == "audio.completed"
