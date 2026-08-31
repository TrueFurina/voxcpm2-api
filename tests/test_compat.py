import os

import pytest

import voxcpm2_api.compat as compat

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def test_prepare_process_environment_sets_macos_thread_limits(monkeypatch) -> None:
    monkeypatch.setattr(compat.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(compat, "_ENV_PREPARED", False)
    for key in (
        "KMP_DUPLICATE_LIB_OK",
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "PYTORCH_ENABLE_MPS_FALLBACK",
    ):
        monkeypatch.delenv(key, raising=False)

    compat.prepare_process_environment()

    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OMP_THREAD_LIMIT"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.0"
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"

def test_sdpa_patch_handles_voxcpm2_forward_step_mask_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    compat.apply_torch_compat_patches()

    query = torch.randn((1, 32, 1, 64), dtype=torch.bfloat16)
    key = torch.randn((1, 8, 16, 64), dtype=torch.bfloat16)
    value = torch.randn((1, 8, 16, 64), dtype=torch.bfloat16)
    attn_mask = torch.arange(16) <= 3

    result = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        enable_gqa=True,
    )

    assert result.shape == (1, 32, 1, 64)
