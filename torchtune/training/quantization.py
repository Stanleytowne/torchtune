# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from torch import nn
from torchtune.modules.peft.lora import LoRALinear, QATLoRALinear


try:
    # torchao 0.7+
    from torchao.dtypes import TensorCoreTiledLayout
except ImportError:
    # torchao 0.6 and before
    from torchao.dtypes import TensorCoreTiledLayoutType as TensorCoreTiledLayout

from torchao.quantization import (
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    quantize_,
)

try:
    # torchao 0.7+
    from torchao.quantization.qat import (
        Int4WeightOnlyQATQuantizer,
        Int8DynActInt4WeightQATQuantizer,
        PissaQuantInt4WeightQATQuantizer,
    )
    from torchao.quantization.qat.linear import (
        disable_4w_fake_quant,
        disable_8da4w_fake_quant,
        disable_pissaquant_fake_quant,
        enable_4w_fake_quant,
        enable_8da4w_fake_quant,
        enable_pissaquant_fake_quant,
    )
except ImportError:
    # torchao 0.6 and before
    from torchao.quantization.prototype.qat import (
        disable_4w_fake_quant,
        disable_8da4w_fake_quant,
        enable_4w_fake_quant,
        enable_8da4w_fake_quant,
        Int4WeightOnlyQATQuantizer,
        Int8DynActInt4WeightQATQuantizer,
    )


__all__ = [
    "get_quantizer_mode",
    "Int4WeightOnlyQuantizer",
    "Int4WeightOnlyQATQuantizer",
    "Int4WeightOnlyQATQuantizerModuleSwap",
    "Int8DynActInt4WeightQuantizer",
    "Int8DynActInt4WeightQATQuantizer",
    "Int8DynActInt4WeightQATQuantizerModuleSwap",
    "PissaQuantInt4WeightQATQuantizer",
    "Int8ActivationOnlyQuantizer",
]


_torchao_0_7_supported = True
try:
    from torchao.quantization import qat  # noqa: F401
except ImportError:
    _torchao_0_7_supported = False

_quantizer_to_mode = {}
_quantizer_mode_to_disable_fake_quant = {}
_quantizer_mode_to_enable_fake_quant = {}


# ========================================
# int8 dynamic activations + int4 weight |
# ========================================


class Int8DynActInt4WeightQuantizer:
    """
    Quantizer for applying int8 per token dynamic activation + int4
    per group weight quantization to linear layers in the model.
    """

    def __init__(self, groupsize: int = 256):
        self.groupsize = groupsize

    def quantize(self, model):
        quantize_fn = int8_dynamic_activation_int4_weight(self.groupsize)
        quantize_(model, quantize_fn)
        return model


_quantizer_to_mode[Int8DynActInt4WeightQuantizer] = "8da4w"
_quantizer_to_mode[Int8DynActInt4WeightQATQuantizer] = "8da4w-qat"
_quantizer_mode_to_disable_fake_quant["8da4w-qat"] = disable_8da4w_fake_quant
_quantizer_mode_to_enable_fake_quant["8da4w-qat"] = enable_8da4w_fake_quant


# ==================
# int4 weight only |
# ==================


class Int4WeightOnlyQuantizer:
    """
    Quantizer for applying int4 per group weight only quantization
    to linear layers in the model using the efficient tinygemm kernel.
    """

    def __init__(self, groupsize: int = 128, inner_k_tiles: int = 8):
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

    def quantize(self, model):
        layout_type = TensorCoreTiledLayout(self.inner_k_tiles)
        quantize_fn = int4_weight_only(self.groupsize, layout_type)
        quantize_(model, quantize_fn)
        return model


_quantizer_to_mode[Int4WeightOnlyQuantizer] = "4w"
_quantizer_to_mode[Int4WeightOnlyQATQuantizer] = "4w-qat"
_quantizer_mode_to_disable_fake_quant["4w-qat"] = disable_4w_fake_quant
_quantizer_mode_to_enable_fake_quant["4w-qat"] = enable_4w_fake_quant

# ==========================
# pissaquant (int4, QAT)   |
# ==========================
if "PissaQuantInt4WeightQATQuantizer" in globals():
    _quantizer_to_mode[PissaQuantInt4WeightQATQuantizer] = "pissaquant-qat"
    _quantizer_mode_to_disable_fake_quant["pissaquant-qat"] = disable_pissaquant_fake_quant
    _quantizer_mode_to_enable_fake_quant["pissaquant-qat"] = enable_pissaquant_fake_quant


# ====================== #
# Backward compatibility #
# ====================== #


# int4 weight-only
class Int4WeightOnlyQATQuantizerModuleSwap(Int4WeightOnlyQATQuantizer):
    pass


disable_4w_fake_quant_module_swap = disable_4w_fake_quant
enable_4w_fake_quant_module_swap = enable_4w_fake_quant
_quantizer_to_mode[Int4WeightOnlyQATQuantizerModuleSwap] = "4w-qat-module-swap"
_quantizer_mode_to_disable_fake_quant[
    "4w-qat-module-swap"
] = disable_4w_fake_quant_module_swap
_quantizer_mode_to_enable_fake_quant[
    "4w-qat-module-swap"
] = enable_4w_fake_quant_module_swap


# int8 dynamic activations + int4 weight
class Int8DynActInt4WeightQATQuantizerModuleSwap(Int8DynActInt4WeightQATQuantizer):
    pass


disable_8da4w_fake_quant_module_swap = disable_8da4w_fake_quant
enable_8da4w_fake_quant_module_swap = enable_8da4w_fake_quant
_quantizer_to_mode[Int8DynActInt4WeightQATQuantizerModuleSwap] = "8da4w-qat-module-swap"
_quantizer_mode_to_disable_fake_quant[
    "8da4w-qat-module-swap"
] = disable_8da4w_fake_quant_module_swap
_quantizer_mode_to_enable_fake_quant[
    "8da4w-qat-module-swap"
] = enable_8da4w_fake_quant_module_swap


# =========================================
# int8 dynamic activation only (no weight) |
# =========================================


import torch


class Int8ActivationOnlyQuantizer:
    """
    Quantizer that applies only int8 per-token dynamic asymmetric activation
    quantization to nn.Linear layers, without modifying weights.

    This is designed for evaluating models whose weights are already quantized
    (e.g., by LoRDS, GPTQ, AWQ) in a W4A8 setting. The activation quantization
    is aligned with torchao's Int8DynActInt4WeightQuantizer:
      - dtype: int8 (signed, -128 to 127)
      - granularity: per-token (each token gets its own scale/zero_point)
      - mapping: asymmetric
      - applied to: all nn.Linear layers (including lm_head, excluding Embedding/Norm)
    """

    def quantize(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_pre_hook(self._int8_activation_hook)
        return model

    @staticmethod
    def _int8_activation_hook(module, input):
        x = input[0]
        # Per-token asymmetric signed int8 fake quantization
        # Aligned with torchao._int8_asymm_per_token_quant:
        #   MappingType.ASYMMETRIC, target_dtype=int8, per_token block_size
        quant_min, quant_max = -128, 127
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        eps = torch.finfo(torch.float32).eps
        scale = ((x_max - x_min) / (quant_max - quant_min)).clamp(min=eps).to(torch.float32)
        zero_point = torch.clamp(
            torch.round(quant_min - x_min / scale), quant_min, quant_max
        ).to(torch.int8)
        x_q = torch.clamp(torch.round(x / scale + zero_point.float()), quant_min, quant_max)
        x_deq = ((x_q - zero_point.float()) * scale).to(x.dtype)
        return (x_deq,) + input[1:]


def get_quantizer_mode(quantizer: Optional[Callable]) -> Optional[str]:
    """Given a quantizer object, returns a string that specifies the type of quantization.

    For example, in the case of int4 weight only quantization, we'll return "4w".
    If the quantizer is not recognized as a known quantizer, we'll return None.

    Currently supported:

    - :class:`~torchtune.training.quantization.Int8DynActInt4WeightQuantizer`: "8da4w"
    - :class:`~torchtune.training.quantization.Int4WeightOnlyQuantizer`: "4w"
    - :class:`~torchao.quantization.qat.Int8DynActInt4WeightQATQuantizer`: "8da4w-qat"
    - :class:`~torchao.quantization.qat.Int4WeightOnlyQATQuantizer`: "4w-qat"

    Args:
        quantizer (Optional[Callable]): A callable object that implements the `quantize` method.

    Returns:
        Optional[str]: The quantization mode.
    """
    return _quantizer_to_mode.get(type(quantizer), None)


def _get_disable_fake_quant(quantizer_mode: str) -> Callable:
    """Given a quantizer mode, return the corresponding function for disabling fake
    quantize in a model prepared by the quantizer.
    If the quantizer is not recognized as a known QAT quantizer, return None.
    """
    return _quantizer_mode_to_disable_fake_quant.get(quantizer_mode, None)


def _get_enable_fake_quant(quantizer_mode: str) -> Callable:
    """Given a quantizer mode, return the corresponding function for enabling fake
    quantize in a model prepared by the quantizer.
    If the quantizer is not recognized as a known QAT quantizer, return None.
    """
    return _quantizer_mode_to_enable_fake_quant.get(quantizer_mode, None)


def swap_lora_linear_with_qat(
    module: nn.Module,
    # TODO: make the types Optional[FakeQuantizeConfig] once we
    # support torchao 0.7+ by default
    activation_qat_config: Optional["FakeQuantizeConfig"] = None,
    weight_qat_config: Optional["FakeQuantizeConfig"] = None,
) -> None:
    """
    Swap all `LoRALinear` in the model with `QATLoRALinear`.

    This is used for combining QAT + LoRA during finetuning. The resulting linear layers
    will apply the following transformation instead:

        x -> fake_quantize(W_frozen) @ fake_quantize(x) + BAx

    Fake quantization here refers to simulating the quantization numerics without actual
    dtype casting, with the goal of providing improved accuracies when the model is
    ultimately quantized after finetuning.

    Args:
        module (nn.Module): The model to swap linear layers on
        activation_qat_config (Optional[FakeQuantizeConfig]): The config for specifying
            how to fake quantize input activations in the base linear layer
        weight_qat_config (Optional[FakeQuantizeConfig]): The config for specifying
            how to fake quantize base linear weights
    """
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            new_linear = QATLoRALinear.from_lora_linear(
                child,
                activation_qat_config,
                weight_qat_config,
            )
            setattr(module, name, new_linear)
        else:
            swap_lora_linear_with_qat(
                child,
                activation_qat_config,
                weight_qat_config,
            )
