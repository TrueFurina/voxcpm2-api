# Changelog

## 0.2.0 - 2026-04-11

- Fixed VoxCPM2 generation on macOS and CPU-only hosts by disabling unstable MPS usage and patching the CPU `scaled_dot_product_attention` path used by VoxCPM2 incremental decoding.
- Added a standalone `voxcpm2-compat` Python artifact so the compatibility wrapper can be reused without the full API package.
- Hardened request validation for base64 audio payloads and converted invalid payloads from generic 500s into explicit 422 validation errors.
- Improved packaging metadata, Docker defaults, local bootstrap behavior, CI, and release automation.
- Added versioned release outputs for the API package, compatibility wrapper, and macOS Tauri desktop application.
