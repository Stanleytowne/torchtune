import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig

from torchtune import config, training, utils


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
        ab_state = torch.load(ab_path, map_location=device)
        if isinstance(ab_state, dict) and "state_dict" in ab_state:
            ab_state = ab_state["state_dict"]
        if not isinstance(ab_state, dict):
            raise ValueError(
                f"AB state dict at {ab_path} must be a dict, got {type(ab_state)}"
            )
        ckpt_dict.update(ab_state)
    model.load_state_dict(ckpt_dict, assign=True)

    # output = model.output
    # weight_quantizer = output.weight_fake_quantizer
    # w = output.weight.data
    # qw = weight_quantizer(output.weight.data)

    # # Manual int4 weight fake-quant (no quantizer API).
    # bsz = 256
    # w_blocks = w.view(w.shape[0], w.shape[1] // bsz, bsz)
    # eps = torch.finfo(w.dtype).eps
    # max_abs = torch.amax(torch.abs(w_blocks), dim=-1, keepdim=True)
    # scale = torch.clamp(max_abs / 7.5, min=eps)

    # mid_point = 8
    # q_uint = torch.round(w_blocks / scale + mid_point).clamp(0, 15)
    # w_blk = ((q_uint - mid_point) * scale).view_as(w)

    # breakpoint()
    # torch.testing.assert_close(qw, w_blk)

    # Gather and save in torchtune format.
    ckpt_dict = {key: value for key, value in ckpt_dict.items() if 'fake_quantizer' not in key}

    for name in ckpt_dict.keys():
        if name.endswith('.weight'):
            mod = model.get_submodule(name[:-len('.weight')])
            if hasattr(mod, "weight_fake_quantizer") and mod.weight_fake_quantizer is not None:
                weight_quantizer = mod.weight_fake_quantizer
                w = ckpt_dict[name]
                qw = weight_quantizer(w)
                ckpt_dict[name] = qw

    file_name = "model-00001-of-00001"

    output_dir = Path(checkpointer._output_dir)
    output_dir.mkdir(exist_ok=True)
    checkpoint_file = Path.joinpath(
        output_dir, f"{file_name}"
    ).with_suffix(".pt")

    torch.save(ckpt_dict, checkpoint_file)


if __name__ == "__main__":
    sys.exit(recipe_main())
