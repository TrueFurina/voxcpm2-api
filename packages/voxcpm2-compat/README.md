# voxcpm2-compat

Standalone compatibility helpers for running VoxCPM2 safely on macOS and CPU-only hosts.

This package exposes:

- `prepare_process_environment()` for early OpenMP and PyTorch process flags
- `apply_torch_compat_patches()` for the macOS MPS disablement and the CPU-safe SDPA wrapper used by VoxCPM2

It is the same compatibility layer shipped inside `voxcpm2-api`, published separately so it can be reused in custom Python integrations that do not need the full API service.
