# Contributing

Thanks for helping with VoxCPM2 API.

## First evening path

Clone, install the CI extras, and run what works without downloading the VoxCPM2 model.

```bash
git clone https://github.com/shiftbloom-studio/voxcpm2-api.git
cd voxcpm2-api
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
pytest
ruff check .
voxcpm2-api
```

In another terminal:

```bash
curl http://localhost:8000/health
```

`VOXCPM2_STARTUP_LOAD_MODEL` defaults to `false`, so the process starts without loading weights. `/health` and `/v1/runtime` work in that mode. Real `/v1/speech` needs the `voxcpm` extra and a model.

`./scripts/bootstrap.sh` installs `.[dev,voxcpm]` when you want that heavier path.

## Optional extras

CI and docs-only changes use `[dev]`. Add the others only when you need the matching runtime:

| Extra | When it is required |
| --- | --- |
| `dev` | pytest, `ruff check .`, packaging |
| `voxcpm` | official backend, prompt or reference audio, real synthesis |
| `nanovllm` | Linux CUDA `nano-vllm-voxcpm` path |
| `asr` | `/v1/transcribe` via faster-whisper |

## Desktop UI

The Tauri client in [`voxcpm2-ui/`](./voxcpm2-ui) is optional for API work.

```bash
cd voxcpm2-ui/src-tauri
cargo tauri dev
```

## Pull requests

- Keep the change focused and small enough to review.
- Run `pytest` and `ruff check .` before opening a PR.
- Use conventional commits (`docs:`, `fix:`, `feat:`, `chore:`, `ci:`).
- Release wheels and tags live under [shiftbloom-studio/voxcpm2-api](https://github.com/shiftbloom-studio/voxcpm2-api), not the old `fabianzimber/voxcpm2-api` remote.

## Security

Report vulnerabilities privately. Do not open public issues or pull requests that include exploit details.

Contributions are licensed under AGPL-3.0-only, the same as the rest of the repository.
