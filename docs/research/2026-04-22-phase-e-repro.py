"""Phase E β kernel repro harness — host-venv iteration, no Docker.

Reproduces the ε epilogue math in pure PyTorch for bit-close validation
of the CuTe kernel output. Reusable across tasks 7-17.
"""
import torch


def epsilon_epilogue_ref(
    residual_post_ln: torch.Tensor,   # [nat, hidden] BF16
    mlp_out: torch.Tensor,            # [nat, hidden] BF16
    next_gamma: torch.Tensor | None,  # [hidden] BF16, None for last layer
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (residual_final, next_hidden_normed_or_residual_if_last).

    When next_gamma is None (last fusion-active layer), the second return
    value is residual_final itself (β kernel's _emit_next_layernorm=False).
    """
    # residual_add happens in FP32 then BF16 cast
    residual_final = (
        residual_post_ln.float() + mlp_out.float()
    ).to(torch.bfloat16)

    if next_gamma is None:
        return residual_final, residual_final

    # next-layer input_layernorm: RMSNorm(residual_final) * next_gamma
    rf32 = residual_final.float()
    variance = rf32.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    normed = ((rf32 * rstd) * (1.0 + next_gamma.float())).to(torch.bfloat16)
    return residual_final, normed


if __name__ == "__main__":
    # Smoke
    nat, hidden = 4, 5120
    residual_post = torch.randn(nat, hidden, dtype=torch.bfloat16, device='cuda')
    mlp_out = torch.randn(nat, hidden, dtype=torch.bfloat16, device='cuda')
    next_gamma = torch.ones(hidden, dtype=torch.bfloat16, device='cuda')
    rf, nh = epsilon_epilogue_ref(residual_post, mlp_out, next_gamma)
    print(f"residual_final: {rf.shape} {rf.dtype}")
    print(f"next_hidden:    {nh.shape} {nh.dtype}")
    # Last-layer case
    rf2, nh2 = epsilon_epilogue_ref(residual_post, mlp_out, None)
    assert torch.equal(nh2, rf2), "last-layer case should return residual_final"
    print("epsilon_epilogue_ref OK")
