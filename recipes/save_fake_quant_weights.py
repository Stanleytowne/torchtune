import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig

from torchtune import config, training, utils


def _enable_fake_quant(mod: torch.nn.Module) -> None:
    if hasattr(mod, "weight_fake_quantizer") and mod.weight_fake_quantizer is not None:
        if hasattr(mod.weight_fake_quantizer, "enable_fake_quant"):
            mod.weight_fake_quantizer.enable_fake_quant(True)


def _apply_fake_quant_inplace(mod: torch.nn.Module) -> None:
    if hasattr(mod, "weight_fake_quantizer") and mod.weight_fake_quantizer is not None:
        if hasattr(mod, "weight"):
            wq = mod.weight_fake_quantizer(mod.weight)
            mod.weight.data = wq.to(dtype=mod.weight.dtype)


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Load a QAT config, apply fake-quantization to weights in-place, and save
    a bfloat16 state_dict whose weights already include fake-quant effects.

    Expected config fields:
      - model, checkpointer, quantizer, output_dir, dtype, device
      - optional: fake_quantized_output_path (str)
      - optional: quantizer.ab_state_dict_path (for pissaquant)
    """
    log = utils.get_logger("INFO")
    if cfg.get("quantizer", None) is None:
        raise ValueError("quantizer must be specified to save fake-quant weights.")

    device = utils.get_device(device=cfg.device)
    dtype = training.get_dtype(cfg.dtype, device=device)

    checkpointer = config.instantiate(cfg.checkpointer, should_load_recipe_state=False)
    checkpoint_dict = checkpointer.load_checkpoint()
    model_state_dict = checkpoint_dict[training.MODEL_KEY]

    # Build model on meta to avoid CPU RAM spikes.
    with training.set_default_dtype(dtype), torch.device("meta"):
        model = config.instantiate(cfg.model)

    quantizer = config.instantiate(cfg.quantizer)
    quantizer.precision = dtype
    model = quantizer.prepare(model)

    # If pissaquant AB parameters are provided, merge them before loading.
    ab_path = getattr(quantizer, "pissaquant_ab_init_path", None)
    if ab_path:
        ab_state = torch.load(ab_path, map_location="cpu")
        if isinstance(ab_state, dict) and "state_dict" in ab_state:
            ab_state = ab_state["state_dict"]
        if not isinstance(ab_state, dict):
            raise ValueError(
                f"AB state dict at {ab_path} must be a dict, got {type(ab_state)}"
            )
        model_state_dict.update(ab_state)

    training.load_from_full_model_state_dict(
        model, model_state_dict, device, strict=True, cpu_offload=False
    )

    # Ensure fake-quant is enabled, then apply it in-place to weights.
    model.apply(_enable_fake_quant)
    with torch.no_grad():
        model.apply(_apply_fake_quant_inplace)

    # Gather and save in torchtune format.
    cpu_state_dict = training.gather_cpu_state_dict(
        model, is_rank_zero=True, device=device
    )

    checkpoint_dict = {}
    checkpoint_dict.update({training.MODEL_KEY: cpu_state_dict})

    checkpointer.save_checkpoint(
        checkpoint_dict,
        epoch=0,
    )


if __name__ == "__main__":
    sys.exit(recipe_main())
