from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOXCPM2_",
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    api_title: str = "VoxCPM2 Service"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    model_id: str = "openbmb/VoxCPM2"
    model_path: str | None = None
    model_cache_dir: str | None = None
    hf_endpoint: str | None = None
    local_files_only: bool = False
    load_denoiser: bool = False
    optimize_model: bool = True
    startup_load_model: bool = False

    prefer_backend: Literal["auto", "voxcpm", "nanovllm"] = "auto"
    allow_voxcpm: bool = True
    allow_nanovllm: bool = True

    default_language: str = "en"
    default_response_format: Literal["wav", "base64"] = "wav"
    default_stream_chunk_format: Literal["pcm16", "wav"] = "pcm16"
    default_normalize_text: bool = False
    default_denoise_conditioning_audio: bool = False
    sample_rate_fallback: int = 48000
    max_text_chars: int = 4000

    nanovllm_devices: list[int] = Field(default_factory=lambda: [0])
    nanovllm_max_num_batched_tokens: int = 8192
    nanovllm_max_num_seqs: int = 16
    nanovllm_max_model_len: int = 4096
    nanovllm_gpu_memory_utilization: float = 0.95

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        if not value:
            return ["*"]
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator("nanovllm_devices", mode="before")
    @classmethod
    def parse_nanovllm_devices(cls, value: str | list[int]) -> list[int]:
        if isinstance(value, list):
            return value
        if not value:
            return [0]
        return [int(item.strip()) for item in value.split(",") if item.strip()]

    @property
    def model_source(self) -> str:
        return self.model_path or self.model_id


@lru_cache
def get_settings() -> Settings:
    return Settings()
