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

        scales_block = _pissaquant_blockwise_symmetric_scales(
            w, block_size=init_block_size, eps=cfg_q.eps
        )
        s_full = scales_block.repeat_interleave(init_block_size, dim=1)
        B, A = _pissaquant_lowrank_factorize(
            s_full, rank=rank, niter=cfg_q.svd_niter
        )

        # Compute quantization errors (Frobenius norm)
        w_fp32 = w.to(torch.float32)
        # PissaQuant (AB) quantization
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

        utils.log_rank_zero(
            log,
            f"Quant error (fro) {module_name}: pissaquant_ab={err_ab:.6e}, blockwise_int4={err_blk:.6e}",
        )

        prefix = weight_key[: -len(".weight")] if weight_key != "weight" else ""
        A_key = f"{prefix}.weight_fake_quantizer.A" if prefix else "weight_fake_quantizer.A"
        B_key = f"{prefix}.weight_fake_quantizer.B" if prefix else "weight_fake_quantizer.B"

        ab_state_dict[A_key] = A.to(dtype=w.dtype)
        ab_state_dict[B_key] = B.to(dtype=w.dtype)

        total_ab += A.numel() + B.numel()
        total_scale += w.shape[0] * (w.shape[1] // cfg_q.block_size)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(cfg.get("pissaquant_ab_init_path", output_dir / "pissaquant_ab_init.pth"))
    torch.save(ab_state_dict, out_path)

    utils.log_rank_zero(
        log,
        f"PissaQuant param count: AB={total_ab} vs int4 scales={total_scale} (block_size={cfg.block_size})",
    )


if __name__ == "__main__":
    sys.exit(recipe_main())
