import os

import pytest

from voxcpm2_api.compat import apply_torch_compat_patches

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

torch = pytest.importorskip("torch")


def test_sdpa_patch_handles_voxcpm2_forward_step_mask_on_cpu() -> None:
    apply_torch_compat_patches()

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
