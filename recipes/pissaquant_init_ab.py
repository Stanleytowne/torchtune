import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig

from torchtune import config, training, utils
from torchtune.recipe_interfaces import FTRecipeInterface


def _resolve_weight_key(
    module_name: str, state_dict: Dict[str, torch.Tensor]
) -> Optional[str]:
    """
    Resolve a weight key from a module name. If activation checkpointing wrappers
    were used, the checkpoint keys may not include `_checkpoint_wrapped_module`.
    """
    checkpoint_wrapper_seg = "._checkpoint_wrapped_module"
    candidate_module_names = []
    if module_name:
        candidate_module_names.append(module_name)
        if checkpoint_wrapper_seg in module_name:
            candidate_module_names.append(module_name.replace(checkpoint_wrapper_seg, ""))
    else:
        candidate_module_names.append("")
    for mn in candidate_module_names:
        wk = f"{mn}.weight" if mn else "weight"
        if wk in state_dict:
            return wk
    return None


def _refine_ab(
    w_fp32: torch.Tensor,
    B: torch.Tensor,
    A: torch.Tensor,
    *,
    eps: float,
    steps: int,
    lr: float,
) -> Dict[str, torch.Tensor]:
    """
    Iteratively refine A/B to reduce quantization error.
    We alternate:
      A) fix A/B, compute nearest int4 Q
      B) fix Q, optimize A/B with MSE loss
    """
    if steps <= 0:
        return {"B": B, "A": A}

    device = w_fp32.device
    B = B.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
    A = A.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([B, A], lr=lr)

    qmin, qmax = -8, 7
    for _ in range(steps):
        with torch.no_grad():
            scale = torch.abs(B @ A) + eps
            scale = torch.clamp(scale, min=eps)
            q = torch.round(w_fp32 / scale).clamp(qmin, qmax)

        optimizer.zero_grad(set_to_none=True)
        scale = torch.abs(B @ A) + eps
        scale = torch.clamp(scale, min=eps)
        w_hat = scale * q
        loss = torch.linalg.norm(w_hat - w_fp32)
        loss.backward()
        optimizer.step()

    return {"B": B.detach(), "A": A.detach()}


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Initialize PissaQuant A/B parameters from full-precision weights and save them
    to a standalone file for later training.

    Expected config fields:
      - model, checkpointer, quantizer, output_dir, dtype, device
      - optional: pissaquant_ab_init_path (str)
    """
    log = utils.get_logger("INFO")

    if cfg.get("quantizer", None) is None:
        raise ValueError("quantizer must be specified to initialize PissaQuant A/B.")

    # Load full-precision weights using the configured checkpointer.
    checkpointer = config.instantiate(cfg.checkpointer, should_load_recipe_state=False)
    checkpoint_dict = checkpointer.load_checkpoint()
    model_state_dict = checkpoint_dict[training.MODEL_KEY]

    # Build model on meta to get module names.
    device = utils.get_device(device=cfg.device)
    dtype = training.get_dtype(cfg.dtype, device=device)
    with training.set_default_dtype(dtype), torch.device("meta"):
        model = config.instantiate(cfg.model)

    quantizer = config.instantiate(cfg.quantizer)
    model = quantizer.prepare(model)

    from torchao.quantization.qat.linear import (
        PissaQuantQATLinear,
        _pissaquant_blockwise_symmetric_scales,
        _pissaquant_lowrank_factorize,
    )

    cfg_q = getattr(quantizer, "weight_qat_config", None)
    if cfg_q is None:
        raise ValueError("quantizer.weight_qat_config is missing; not a pissaquant quantizer?")

    ab_state_dict: Dict[str, torch.Tensor] = {}
    total_ab = 0
    total_scale = 0

    refine_steps = int(cfg.get("pissaquant_ab_refine_steps", 0))
    refine_lr = float(cfg.get("pissaquant_ab_refine_lr", 1e-3))
    compute_device = torch.device(cfg.get("pissaquant_ab_device", cfg.device))
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        utils.log_rank_zero(
            log,
            "pissaquant_ab_device was set to cuda but CUDA is not available; falling back to CPU.",
        )
        compute_device = torch.device("cpu")
    utils.log_rank_zero(log, f"PissaQuant AB init device: {compute_device}")

    for module_name, mod in model.named_modules():
        if not isinstance(mod, PissaQuantQATLinear):
            continue

        weight_key = _resolve_weight_key(module_name, model_state_dict)
        if weight_key is None:
            raise KeyError(f"Missing weight key for module '{module_name}' in checkpoint.")
        w = model_state_dict[weight_key]
        if w.dim() != 2:
            continue

        rank = cfg_q.compute_rank(in_features=w.shape[1], out_features=w.shape[0])
        init_block_size = w.shape[1] // rank
        while w.shape[1] % init_block_size != 0:
            init_block_size -= 1
        if init_block_size <= 0:
            raise ValueError(
                f"init_block_size must be > 0 for {module_name}, "
                f"got in_features={w.shape[1]}, rank={rank}."
            )

        # Move weight to compute device for factorization and error measurement
        w_fp32 = w.to(device=compute_device, dtype=torch.float32)
        scales_block = _pissaquant_blockwise_symmetric_scales(
            w_fp32, block_size=init_block_size, eps=cfg_q.eps
        )
        s_full = scales_block.repeat_interleave(init_block_size, dim=1)
        B, A = _pissaquant_lowrank_factorize(
            s_full, rank=rank, niter=cfg_q.svd_niter
        )

        # Compute quantization errors (Frobenius norm)
        # PissaQuant (AB) quantization before refinement
        scale_ab = torch.abs(B.to(torch.float32) @ A.to(torch.float32)) + float(
            cfg_q.eps
        )
        q_ab = torch.round(w_fp32 / scale_ab).clamp(-8, 7)
        w_ab = q_ab * scale_ab
        err_ab = torch.linalg.norm(w_fp32 - w_ab).item()

        # Block-wise int4 weight-only quantization (baseline)
        bsz = cfg_q.block_size
        if w.shape[1] % bsz != 0:
            raise ValueError(
                f"in_features ({w.shape[1]}) must be divisible by block_size ({bsz}) "
                "to compute block-wise baseline error."
            )
        w_blocks = w_fp32.view(w.shape[0], w.shape[1] // bsz, bsz)
        scale_blk = torch.amax(torch.abs(w_blocks), dim=-1, keepdim=True) / 7.0
        scale_blk = torch.clamp(scale_blk, min=float(cfg_q.eps))
        q_blk = torch.round(w_blocks / scale_blk).clamp(-8, 7)
        w_blk = (q_blk * scale_blk).view_as(w_fp32)
        err_blk = torch.linalg.norm(w_fp32 - w_blk).item()

        # Optional refinement
        if refine_steps > 0:
            refined = _refine_ab(
                w_fp32,
                B,
                A,
                eps=float(cfg_q.eps),
                steps=refine_steps,
                lr=refine_lr,
            )
            B = refined["B"]
            A = refined["A"]

            scale_ab = torch.abs(B @ A) + float(cfg_q.eps)
            q_ab = torch.round(w_fp32 / scale_ab).clamp(-8, 7)
            w_ab = q_ab * scale_ab
            err_ab_refined = torch.linalg.norm(w_fp32 - w_ab).item()
            utils.log_rank_zero(
                log,
                (
                    f"Quant error (fro) {module_name}: "
                    f"pissaquant_ab={err_ab:.6e} -> {err_ab_refined:.6e}, "
                    f"blockwise_int4={err_blk:.6e}"
                ),
            )
        else:
            utils.log_rank_zero(
                log,
                f"Quant error (fro) {module_name}: pissaquant_ab={err_ab:.6e}, blockwise_int4={err_blk:.6e}",
            )

        prefix = weight_key[: -len(".weight")] if weight_key != "weight" else ""
        if 'layers' in prefix:
            prefix = prefix + '._checkpoint_wrapped_module'
        A_key = f"{prefix}.weight_fake_quantizer.A" if prefix else "weight_fake_quantizer.A"
        B_key = f"{prefix}.weight_fake_quantizer.B" if prefix else "weight_fake_quantizer.B"

        # Store on CPU for saving; preserve original weight dtype
        ab_state_dict[A_key] = A.to(device="cpu", dtype=w.dtype)
        ab_state_dict[B_key] = B.to(device="cpu", dtype=w.dtype)

        total_ab += A.numel() + B.numel()
        total_scale += w.shape[0] * (w.shape[1] // cfg_q.block_size)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(cfg.get("pissaquant_ab_init_path", output_dir / "pissaquant_ab_init.pth"))
    torch.save(ab_state_dict, out_path)

    utils.log_rank_zero(
        log,
        f"PissaQuant param count: AB={total_ab} vs int4 scales={total_scale} (block_size={cfg_q.block_size})",
    )


if __name__ == "__main__":
    sys.exit(recipe_main())
