from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)
    language: str = "en"
    prompt_text: str | None = None
    prompt_audio_path: str | None = None
    prompt_audio_base64: str | None = None
    reference_audio_path: str | None = None
    reference_audio_base64: str | None = None
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    min_len: int = 2
    max_len: int = 4096
    normalize_text: bool | None = None
    denoise_conditioning_audio: bool | None = None
    response_format: Literal["wav", "base64"] = "wav"

    @model_validator(mode="after")
    def validate_audio_sources(self) -> "SynthesisRequest":
        pairs = [
            (self.prompt_audio_path, self.prompt_audio_base64, "prompt audio"),
            (self.reference_audio_path, self.reference_audio_base64, "reference audio"),
        ]
        for path_value, base64_value, label in pairs:
            if path_value and base64_value:
                raise ValueError(f"Provide either {label} path or base64, not both")
        if self.prompt_audio_path and not self.prompt_text:
            raise ValueError("prompt_text is required when prompt audio is provided")
        if self.prompt_audio_base64 and not self.prompt_text:
            raise ValueError("prompt_text is required when prompt audio is provided")
        return self

    @property
    def has_conditioning(self) -> bool:
        return any(
            [
                self.prompt_audio_path,
                self.prompt_audio_base64,
                self.reference_audio_path,
                self.reference_audio_base64,
            ]
        )


class StreamingSynthesisRequest(SynthesisRequest):
    chunk_format: Literal["pcm16", "wav"] = "pcm16"


class RuntimeResponse(BaseModel):
    selected_backend: str
    requested_backend: str
    model_source: str
    hardware: dict[str, object]
    backend_status: dict[str, dict[str, object]]


class SynthesisResponse(BaseModel):
    backend: str
    device: str
    sample_rate: int
    audio_base64: str
    language: str


class ErrorEnvelope(BaseModel):
    error: str
    detail: str | None = None
