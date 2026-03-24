# Copyright (c) 2025, Shaanxi Yuanyi Intelligent Technology Co., Ltd.
# This file is part of a project licensed under the MIT License.
# It is developed based on the MoCo project by Meta Platforms, Inc.
# Original MoCo repository: https://github.com/facebookresearch/moco
#
# This project includes significant modifications tailored for SAR land-cover classification,
# including the design of domain-specific modules and the use of large-scale SAR datasets
# to improve performance and generalization on downstream SAR tasks.


import os
import random
from functools import wraps
from typing import List, NamedTuple

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter, Normalize, RandomGrayscale, ToTensor


# =============================================================================
# Data Structures
# =============================================================================

class ImageWithTransInfo(NamedTuple):
    """Bundles an augmented image with the spatial transform metadata."""
    image:  torch.Tensor  # augmented image tensor
    transf: List          # crop coords [y, x, h, w] + flipped flag
    ratio:  List          # resize ratio relative to original image [rh, rw]
    size:   List          # original image size (h, w)


# =============================================================================
# Transform-Info Plumbing
# =============================================================================

def free_pass_trans_info(func):
    """Wrap a standard transform so it passes transf/ratio through unchanged."""
    @wraps(func)
    def decorator(img, transf, ratio):
        return func(img), transf, ratio
    return decorator


def _with_trans_info(transform):
    """Return the with_trans_info variant of a transform, or wrap its __call__."""
    if hasattr(transform, 'with_trans_info'):
        return transform.with_trans_info
    return free_pass_trans_info(transform)


def _get_size(size):
    """Normalise size to (h, w)."""
    if isinstance(size, int):
        return size, size
    return size


def _update_transf_and_ratio(transf_global, ratio_global,
                              transf_local=None, ratio_local=None):
    """Compose a local crop/resize transform into the running global transform."""
    if transf_local:
        i_global, j_global, *_ = transf_global
        i_local, j_local, h_local, w_local = transf_local
        transf_global = [
            int(round(i_local / ratio_global[0] + i_global)),
            int(round(j_local / ratio_global[1] + j_global)),
            int(round(h_local / ratio_global[0])),
            int(round(w_local / ratio_global[1])),
        ]
    if ratio_local:
        ratio_global = [g * l for g, l in zip(ratio_global, ratio_local)]
    return transf_global, ratio_global


# =============================================================================
# Transforms with Spatial Tracking
# =============================================================================

class Compose:
    """Drop-in replacement for torchvision Compose that can track spatial transforms."""

    def __init__(self, transforms, with_trans_info: bool = False, seed=None):
        self.transforms     = transforms
        self.with_trans_info = with_trans_info
        self.seed           = seed

    def __call__(self, img):
        if self.with_trans_info:
            return self._call_with_trans_info(img)
        return self._call_default(img)

    def _call_default(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def _call_with_trans_info(self, img):
        w, h   = img.size
        transf = [0, 0, h, w]
        ratio  = [1., 1.]

        for t in self.transforms:
            t_fn = _with_trans_info(t)
            try:
                if self.seed:
                    random.seed(self.seed)
                    torch.manual_seed(self.seed)
                img, transf, ratio = t_fn(img, transf, ratio)
            except Exception as e:
                raise Exception(f'{e}: from {t.__self__}')

        return ImageWithTransInfo(img, transf, ratio, (h, w))


class CenterCrop(transforms.CenterCrop):
    def with_trans_info(self, img, transf, ratio):
        w, h   = img.size
        oh, ow = _get_size(self.size)
        transf_local = [int(round((w - ow) * 0.5)), int(round((h - oh) * 0.5)), oh, ow]
        transf, ratio = _update_transf_and_ratio(transf, ratio, transf_local, None)
        return F.center_crop(img, self.size), transf, ratio


class Resize(transforms.Resize):
    def with_trans_info(self, img, transf, ratio):
        w, h        = img.size
        resized     = F.resize(img, self.size, self.interpolation)
        ow, oh      = resized.size
        ratio_local = [oh / h, ow / w]
        transf, ratio = _update_transf_and_ratio(transf, ratio, None, ratio_local)
        return resized, transf, ratio


class RandomResizedCrop(transforms.RandomResizedCrop):
    def with_trans_info(self, img, transf, ratio):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        oh, ow     = _get_size(self.size)
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, [i, j, h, w], [oh / h, ow / w]
        )
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), transf, ratio


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def with_trans_info(self, img, transf, ratio):
        if torch.rand(1) < self.p:
            transf.append(True)
            return F.hflip(img), transf, ratio
        transf.append(False)
        return img, transf, ratio


class RandomOrder(transforms.RandomOrder):
    def with_trans_info(self, img, transf, ratio):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img, transf, ratio = _with_trans_info(self.transforms[i])(img, transf, ratio)
        return img, transf, ratio


class RandomApply(transforms.RandomApply):
    def with_trans_info(self, img, transf, ratio):
        if self.p < random.random():
            return img, transf, ratio
        for t in self.transforms:
            img, transf, ratio = _with_trans_info(t)(img, transf, ratio)
        return img, transf, ratio


# =============================================================================
# Augmentation Primitives
# =============================================================================

class GaussianBlur:
    """Gaussian blur augmentation from SimCLR (https://arxiv.org/abs/2002.05709)."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class Solarize:
    def __init__(self, threshold: float):
        assert 0 < threshold < 1
        self.threshold = round(threshold * 256)

    def __call__(self, img):
        return ImageOps.solarize(img, self.threshold)

    def __repr__(self):
        return f"{self.__class__.__name__}(threshold={self.threshold})"


# =============================================================================
# Dataset & Collation
# =============================================================================

class TwoCropsTransform:
    """Apply the same base transform twice to produce a (query, key) pair."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


class SAR_dataset(Dataset):
    """Single-channel SAR image dataset; images are replicated to 3-channel RGB."""

    def __init__(self, image_dir: str, transform=None):
        self.transform   = transform
        self.image_paths = [
            os.path.join(image_dir, name) for name in os.listdir(image_dir)
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx])
        img = Image.merge('RGB', [img, img, img])  # single-channel → 3-channel
        return self.transform(img)


def decompose_collated_batch(collated_batch):
    """Unzip a collated batch of ImageWithTransInfo into four parallel lists."""
    if isinstance(collated_batch, ImageWithTransInfo):
        collated_batch = [collated_batch]

    batch_views, batch_transf, batch_ratio, batch_size = [], [], [], []

    for x in collated_batch:
        transf = torch.cat(x.transf).reshape(len(x.transf), x.image.size(0))
        transf = transf.t()  # (batch, num_transform_params)
        batch_views.append(x.image)
        batch_transf.append(transf)
        batch_ratio.append(x.ratio)
        batch_size.append(x.size)

    return batch_views, batch_transf, batch_ratio, batch_size

