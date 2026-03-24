# Copyright (c) 2025, Shaanxi Yuanyi Intelligent Technology Co., Ltd.
# This file is part of a project licensed under the MIT License.
# It is developed based on the MoCo project by Meta Platforms, Inc.
# Original MoCo repository: https://github.com/facebookresearch/moco
#
# This project includes significant modifications tailored for SAR land-cover classification,
# including the design of domain-specific modules and the use of large-scale SAR datasets
# to improve performance and generalization on downstream SAR tasks.


import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
from typing import Dict

from torch import Tensor


# =============================================================================
# Feature Extraction Utility
# =============================================================================

class IntermediateLayerGetter(nn.ModuleDict):
    """Extract intermediate layer outputs from a model by name.

    Strips all layers after the last requested layer to avoid unnecessary
    computation. Only supports top-level submodules (e.g. model.layer3,
    not model.layer3.conv1).

    Args:
        model:         source model whose children will be iterated
        return_layers: mapping from original layer name → output dict key
    """

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset(name for name, _ in model.named_children()):
            raise ValueError("return_layers are not present in model")

        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        remaining     = dict(return_layers)  # consumed as layers are found

        # Keep only the layers up to (and including) the last requested one
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            remaining.pop(name, None)
            if not remaining:
                break

        super().__init__(layers)
        self.return_layers = return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
        return out


# =============================================================================
# Projection Heads
# =============================================================================

class TwoLayerLinearHead(nn.Module):
    """Two-layer MLP projection head: Linear → ReLU → Linear."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 batch_norm: bool = False):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers += [nn.ReLU(), nn.Linear(hidden_size, output_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# =============================================================================
# Encoder
# =============================================================================

class CustomResNet(nn.Module):
    """ResNet encoder with dual-scale output (global + low-level features).

    Taps into layer3 for low-level local features and layer4 for global
    features, each projected to a common embedding dimension.

    Args:
        base_encoder: a torchvision ResNet constructor (e.g. models.resnet50)
        dim:          output embedding dimension for both projection heads
    """

    _RETURN_LAYERS = {'layer3': 'low_feature', 'layer4': 'global_feature'}

    def __init__(self, base_encoder, dim: int = 128):
        super().__init__()
        resnet = base_encoder(pretrained=True)

        self.encoder = IntermediateLayerGetter(resnet, self._RETURN_LAYERS)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.low_projector    = TwoLayerLinearHead(1024, 2048, dim)
        self.global_projector = TwoLayerLinearHead(2048, 2048, dim)

    def forward(self, x: Tensor):
        features = self.encoder(x)

        low    = torch.flatten(self.avgpool(features['low_feature']), 1)
        low    = self.low_projector(low)

        glob   = torch.flatten(self.avgpool(features['global_feature']), 1)
        glob   = self.global_projector(glob)

        return glob, low, features['global_feature']
    