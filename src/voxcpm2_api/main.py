from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from voxcpm2_api.audio import waveform_chunk_to_base64, waveform_to_base64, waveform_to_wav_bytes
from voxcpm2_api.config import Settings, get_settings
from voxcpm2_api.runtime.factory import RuntimeOrchestrator, configure_environment
from voxcpm2_api.schemas import (
    ErrorEnvelope,
    RuntimeResponse,
    StreamingSynthesisRequest,
    SynthesisRequest,
    SynthesisResponse,
)
from voxcpm2_api.service import prepare_audio_assets


def create_app(
    settings: Settings | None = None,
    orchestrator: RuntimeOrchestrator | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_environment(resolved_settings)
        runtime = orchestrator or RuntimeOrchestrator(resolved_settings)
        app.state.settings = resolved_settings
        app.state.runtime = runtime
        if resolved_settings.startup_load_model:
            await runtime.warmup()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title=resolved_settings.api_title, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, object]:
        runtime: RuntimeOrchestrator = app.state.runtime
        snapshot = runtime.runtime_snapshot()
        return {"status": "ok", "runtime": snapshot.model_dump()}

    @app.get("/v1/runtime", response_model=RuntimeResponse)
    async def runtime() -> RuntimeResponse:
        return app.state.runtime.runtime_snapshot()

    @app.post("/v1/speech", responses={503: {"model": ErrorEnvelope}})
    async def synthesize(request: SynthesisRequest):
        settings: Settings = app.state.settings
        if len(request.text) > settings.max_text_chars:
            return JSONResponse(
                status_code=422,
                content=ErrorEnvelope(
                    error="validation_error", detail="text exceeds configured limit"
                ).model_dump(),
            )

        assets = prepare_audio_assets(request)
        try:
            result = await app.state.runtime.synthesize(request, assets)
        except RuntimeError as exc:
            return JSONResponse(
                status_code=503,
                content=ErrorEnvelope(error="runtime_unavailable", detail=str(exc)).model_dump(),
            )
        finally:
            assets.cleanup()

        if request.response_format == "base64":
            return SynthesisResponse(
                backend=result.backend,
                device=result.device,
                sample_rate=result.sample_rate,
                audio_base64=waveform_to_base64(result.waveform, result.sample_rate),
                language=request.language,
            )

        return Response(
            content=waveform_to_wav_bytes(result.waveform, result.sample_rate),
            media_type="audio/wav",
            headers={
                "X-VoxCPM-Backend": result.backend,
                "X-VoxCPM-Device": result.device,
                "X-VoxCPM-Sample-Rate": str(result.sample_rate),
            },
        )

    @app.websocket("/v1/stream")
    async def websocket_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            payload = await websocket.receive_json()
            request = StreamingSynthesisRequest.model_validate(payload)
            assets = prepare_audio_assets(request)
            try:
                runtime_snapshot = app.state.runtime.runtime_snapshot(request)
                await websocket.send_json(
                    {
                        "type": "session.started",
                        "backend": runtime_snapshot.selected_backend,
                        "model_source": runtime_snapshot.model_source,
                    }
                )
                async for chunk in app.state.runtime.stream(request, assets):
                    await websocket.send_json(
                        {
                            "type": "audio.chunk",
                            "sequence": chunk.sequence,
                            "backend": chunk.backend,
                            "device": chunk.device,
                            "sample_rate": chunk.sample_rate,
                            "chunk_format": request.chunk_format,
                            "audio_base64": waveform_chunk_to_base64(
                                chunk.waveform,
                                request.chunk_format,
                                chunk.sample_rate,
                            ),
                        }
                    )
                await websocket.send_json({"type": "audio.completed"})
            finally:
                assets.cleanup()
        except ValidationError as exc:
            await websocket.send_json(
                {"type": "error", "error": "validation_error", "detail": str(exc)}
            )
        except RuntimeError as exc:
            await websocket.send_json(
                {"type": "error", "error": "runtime_unavailable", "detail": str(exc)}
            )
        except WebSocketDisconnect:
            return
        finally:
            await websocket.close()

    return app


app = create_app()
