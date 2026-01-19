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
    device = utils.get_device(device=cfg.device)
    dtype = training.get_dtype(cfg.dtype, device=device)

    quantizer = config.instantiate(cfg.quantizer)
    quantization_mode = training.get_quantizer_mode(quantizer)

    # Load checkpoint
    checkpointer = config.instantiate(cfg.checkpointer)

    # Initialize model
    with training.set_default_dtype(dtype), device:
        model = config.instantiate(cfg.model)

    model = quantizer.prepare(model)
    model = model.to(device=device, dtype=dtype)
    ckpt_dict = checkpointer.load_checkpoint()[
        training.MODEL_KEY
    ]
    for k, v in ckpt_dict.items():
        ckpt_dict[k] = v.to(device)

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
        ckpt_dict.update(ab_state)
    model.load_state_dict(ckpt_dict, assign=True)

    # Ensure fake-quant is enabled, then apply it in-place to weights.
    model.apply(_enable_fake_quant)
    with torch.no_grad():
        model.apply(_apply_fake_quant_inplace)

    # Gather and save in torchtune format.
    ckpt_dict = model.state_dict()

    file_name = "model-00001-of-00001"

    output_dir = Path(checkpointer._output_dir)
    output_dir.mkdir(exist_ok=True)
    checkpoint_file = Path.joinpath(
        output_dir, f"{file_name}"
    ).with_suffix(".pt")

    torch.save(ckpt_dict, checkpoint_file)


if __name__ == "__main__":
    sys.exit(recipe_main())
