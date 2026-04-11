# VoxCPM2 API

Cross-platform FastAPI service for VoxCPM2 with:

- REST synthesis endpoint for full WAV responses
- WebSocket streaming endpoint for low-latency chunk delivery
- automatic backend selection for Linux CUDA, Apple Silicon, and CPU-only hosts
- optional Nano-vLLM acceleration on Linux + NVIDIA GPUs
- request fields that already leave room for prompt continuation, reference audio, and future multilingual expansion

## Backend strategy

The service keeps one API shape and swaps runtimes underneath it:

- **Linux + NVIDIA CUDA** → prefers `nano-vllm-voxcpm` for plain text synthesis
- **Windows / macOS / generic Linux** → uses the official `voxcpm` Python package
- **Prompt / reference audio requests** → always route to the official `voxcpm` backend because that public API already exposes `prompt_wav_path`, `prompt_text`, and `reference_wav_path`

Apple Silicon support currently means **PyTorch MPS**, not ANE. The public ANE project does not support VoxCPM2 yet, so this service does not pretend otherwise.

## Quickstart

### Local dev

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,voxcpm]"
cp .env.example .env
voxcpm2-api
```

Use Python **3.10-3.12**. VoxCPM2 does not support 3.13+ yet.

The helper script does the same bootstrapping:

```bash
./scripts/bootstrap.sh
./scripts/run-dev.sh
```

### Linux CUDA fast path

Install both optional runtimes if you want Nano-vLLM auto-selection:

```bash
pip install -e ".[voxcpm,nanovllm]"
```

Nano-vLLM still requires its upstream CUDA prerequisites such as `flash-attn`.

### Docker

```bash
cp .env.example .env
docker compose up --build
```

## Configuration

All runtime config is environment-driven.

Important variables:

- `VOXCPM2_MODEL_ID` — default `openbmb/VoxCPM2`
- `VOXCPM2_MODEL_PATH` — local model directory override
- `VOXCPM2_PREFER_BACKEND` — `auto`, `voxcpm`, or `nanovllm`
- `VOXCPM2_LOAD_DENOISER` — enable official denoiser loading
- `VOXCPM2_LOCAL_FILES_ONLY` — force offline model loading
- `VOXCPM2_HF_ENDPOINT` — custom Hugging Face mirror endpoint
- `VOXCPM2_NANOVLLM_DEVICES` — comma-separated CUDA device ids

## API

### Health and runtime inspection

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/runtime
```

### REST synthesis

Return WAV directly:

```bash
curl -X POST http://localhost:8000/v1/speech \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from VoxCPM2","response_format":"wav"}' \
  --output out.wav
```

Return JSON with base64 audio:

```bash
curl -X POST http://localhost:8000/v1/speech \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from VoxCPM2","response_format":"base64"}'
```

Prompt continuation request shape:

```json
{
  "text": "Continue this in the same voice.",
  "prompt_text": "Earlier context",
  "prompt_audio_base64": "<wav-as-base64>",
  "reference_audio_base64": "<optional-reference-wav>",
  "response_format": "base64"
}
```

### WebSocket streaming

Connect to `/v1/stream`, send one JSON request, then read `session.started`, repeated `audio.chunk`, and final `audio.completed` frames.

Example request:

```json
{
  "text": "Stream this sentence.",
  "chunk_format": "pcm16"
}
```

## Testing

```bash
. .venv/bin/activate
pytest
```

## Notes for agent integration

- WAV REST responses are easiest for standard TTS clients.
- JSON/base64 REST responses are easier for browser agents and tool-driven orchestrators.
- WebSocket chunk frames include backend, device, sample rate, and chunk sequence so a realtime client can adapt buffering logic without out-of-band metadata.
