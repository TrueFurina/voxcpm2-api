from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from voxcpm2_api.audio import decode_base64_audio
from voxcpm2_api.schemas import SynthesisRequest

_MAX_AUDIO_BYTES = 50 * 1024 * 1024  # 50 MB decoded limit


@dataclass(slots=True)
class PreparedAudioAssets:
    prompt_audio_path: str | None = None
    reference_audio_path: str | None = None
    temp_paths: list[Path] = field(default_factory=list)

    def cleanup(self) -> None:
        for path in self.temp_paths:
            path.unlink(missing_ok=True)


def prepare_audio_assets(request: SynthesisRequest) -> PreparedAudioAssets:
    assets = PreparedAudioAssets(
        prompt_audio_path=request.prompt_audio_path,
        reference_audio_path=request.reference_audio_path,
    )

    if request.prompt_audio_base64:
        assets.prompt_audio_path = _write_temp_wav(request.prompt_audio_base64, assets.temp_paths)
    if request.reference_audio_base64:
        assets.reference_audio_path = _write_temp_wav(
            request.reference_audio_base64, assets.temp_paths
        )
    return assets


def _write_temp_wav(raw_base64: str, temp_paths: list[Path]) -> str:
    payload = decode_base64_audio(raw_base64)
    if len(payload) > _MAX_AUDIO_BYTES:
        raise ValueError(
            f"Decoded audio exceeds the maximum allowed size of {_MAX_AUDIO_BYTES} bytes"
        )
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        temp_file.write(payload)
        temp_file.flush()
    finally:
        temp_file.close()
    temp_path = Path(temp_file.name)
    temp_paths.append(temp_path)
    return str(temp_path)
