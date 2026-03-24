# Copyright (c) 2025, Shaanxi Yuanyi Intelligent Technology Co., Ltd.
# This file is part of a project licensed under the MIT License.
# It is developed based on the MoCo project by Meta Platforms, Inc.
# Original MoCo repository: https://github.com/facebookresearch/moco
#
# This project includes significant modifications tailored for SAR land-cover classification,
# including the design of domain-specific modules and the use of large-scale SAR datasets
# to improve performance and generalization on downstream SAR tasks.

import numpy as np
import torch


# =============================================================================
# Box Utilities
# =============================================================================

def rand_int(low: int, high: int) -> int:
    """Sample a random integer in [low, high)."""
    return np.random.randint(low, high)


def bbox_iou(boxA, boxB) -> float:
    """Compute Intersection over Union between two boxes [l, t, r, b]."""
    inter_l = max(boxA[0], boxB[0])
    inter_t = max(boxA[1], boxB[1])
    inter_r = min(boxA[2], boxB[2])
    inter_b = min(boxA[3], boxB[3])

    inter_area = max(inter_r - inter_l, 0) * max(inter_b - inter_t, 0)
    if inter_area == 0:
        return 0.0

    area_a = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    area_b = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter_area / float(area_a + area_b - inter_area)


def clip_box(box_with_idx: list, input_size: int) -> list:
    """Clamp box coordinates [idx, l, t, r, b] to [0, input_size]."""
    box_with_idx[1] = float(max(0, box_with_idx[1]))
    box_with_idx[2] = float(max(0, box_with_idx[2]))
    box_with_idx[3] = float(min(input_size, box_with_idx[3]))
    box_with_idx[4] = float(min(input_size, box_with_idx[4]))
    return box_with_idx


def jitter_box(box_l, box_t, box_r, box_b, jitter_ratio: float):
    """Apply random scale jitter to a box, returning (l, t, r, b)."""
    box_w = box_r - box_l
    box_h = box_b - box_t
    j = np.random.uniform(1.0 - jitter_ratio, 1.0 + jitter_ratio, size=4)
    new_l = float(box_l + box_w * (j[0] - 1))
    new_t = float(box_t + box_h * (j[1] - 1))
    new_r = float(new_l + box_w * j[2])
    new_b = float(new_t + box_h * j[3])
    return new_l, new_t, new_r, new_b


# =============================================================================
# Box Generator
# =============================================================================

class BoxGenerator:
    """Generate spatially consistent box pairs across two augmented views.

    For each image in the batch, boxes are sampled within the intersection
    region of the two crops so that box1 and box2 correspond to the same
    spatial location in the original image.

    Reference: https://github.com/kakaobrain/scrl/issues/2
    """

    MAX_SAMPLE_TRIES = 50  # max attempts to satisfy the IoU diversity constraint

    def __init__(self, input_size: int, min_size: int, num_patches_per_image: int,
                 box_jittering: bool, box_jittering_ratio: float,
                 iou_threshold: float, grid_based_box_gen: bool):
        self.input_size          = input_size
        self.min_size            = min_size
        self.num_patches         = num_patches_per_image
        self.box_jittering       = box_jittering
        self.box_jittering_ratio = box_jittering_ratio
        self.iou_threshold       = iou_threshold
        self.grid_based          = grid_based_box_gen

    @classmethod
    def init_from_config(cls, cfg):
        return cls(
            input_size          = cfg.augment.input_size,
            min_size            = cfg.network.scrl.min_size,
            num_patches_per_image = cfg.network.scrl.num_patches_per_image,
            box_jittering       = cfg.network.scrl.box_jittering,
            box_jittering_ratio = cfg.network.scrl.jittering_ratio,
            iou_threshold       = cfg.network.scrl.iou_threshold,
            grid_based_box_gen  = cfg.network.scrl.grid_based_box_gen,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, transf) -> list[torch.Tensor]:
        """Return [boxes_view1, boxes_view2], each a (N*num_patches, 5) tensor.

        Each row is [batch_idx, l, t, r, b]. View-2 batch indices are offset
        by the batch size so they can index into a concatenated feature map.
        """
        assert len(transf) != 0, "Transform info must not be empty."

        n_samples = transf[0].size(0)
        boxes1, boxes2 = [], []

        for batch_idx, (t1, t2) in enumerate(zip(transf[0], transf[1])):
            pairs = self._generate_pairs_for_image(t1, t2, batch_idx, n_samples)
            for b1, b2 in pairs:
                boxes1.append(clip_box(b1, self.input_size))
                boxes2.append(clip_box(b2, self.input_size))

        return [torch.tensor(boxes1), torch.tensor(boxes2)]

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _generate_pairs_for_image(self, t1, t2, batch_idx: int, n_samples: int):
        """Yield (box1, box2) pairs for a single image, or [] if no valid overlap."""
        # t layout: [y, x, h, w, flipped]
        int_l = max(t1[1], t2[1]).item()
        int_r = min(t1[1] + t1[3], t2[1] + t2[3]).item()
        int_t = max(t1[0], t2[0]).item()
        int_b = min(t1[0] + t1[2], t2[0] + t2[2]).item()

        # No overlap between the two crops — skip this image
        if int_l >= int_r or int_t >= int_b:
            return []

        scales = self._compute_scales(t1, t2)
        int_w_scaled = round((int_r - int_l) * scales["wmin"])
        int_h_scaled = round((int_b - int_t) * scales["hmin"])

        # Intersection too small to fit even one patch
        if self.min_size >= int_w_scaled or self.min_size >= int_h_scaled:
            return []

        pairs = []
        for patch_idx in range(self.num_patches):
            box = self._sample_box_with_retry(
                int_w_scaled, int_h_scaled, patch_idx, pairs
            )
            b1, b2 = self._project_to_views(
                box, int_l, int_t, t1, t2, scales, batch_idx, n_samples
            )
            pairs.append((b1, b2))
        return pairs

    def _compute_scales(self, t1, t2) -> dict:
        """Precompute per-view and shared min scales."""
        sw1 = self.input_size / t1[3].item()
        sh1 = self.input_size / t1[2].item()
        sw2 = self.input_size / t2[3].item()
        sh2 = self.input_size / t2[2].item()
        wmin = min(sw1, sw2)
        hmin = min(sh1, sh2)
        return dict(sw1=sw1, sh1=sh1, sw2=sw2, sh2=sh2,
                    wmin=wmin, hmin=hmin,
                    wmin_inv=1.0 / wmin, hmin_inv=1.0 / hmin)

    def _sample_box_with_retry(self, int_w: int, int_h: int,
                                patch_idx: int, accepted: list) -> tuple:
        """Sample a (x, y, w, h) box, retrying until IoU constraint is met."""
        for _ in range(self.MAX_SAMPLE_TRIES):
            box = self._sample_box(int_w, int_h)
            if patch_idx == 0 or self.iou_threshold == 1.0:
                break
            if self._satisfies_iou_constraint(box, accepted):
                break
        return box

    def _sample_box(self, int_w: int, int_h: int) -> tuple:
        """Draw a single (x, y, w, h) box within the scaled intersection."""
        if self.grid_based:
            div_w = rand_int(1, int(int_w / self.min_size) + 1)
            div_h = rand_int(1, int(int_h / self.min_size) + 1)
            grid_w = int_w / div_w
            grid_h = int_h / div_h
            gx = rand_int(0, div_w)
            gy = rand_int(0, div_h)
            bw = rand_int(self.min_size, int(grid_w) + 1)
            bh = rand_int(self.min_size, int(grid_h) + 1)
            bx = rand_int(0, int(grid_w - bw) + 1) + int(gx * grid_w)
            by = rand_int(0, int(grid_h - bh) + 1) + int(gy * grid_h)
        else:
            bw = rand_int(self.min_size, int_w)
            bh = rand_int(self.min_size, int_h)
            bx = rand_int(0, int_w - bw)
            by = rand_int(0, int_h - bh)
        return bx, by, bw, bh

    def _satisfies_iou_constraint(self, box: tuple, accepted: list) -> bool:
        """Return True if box overlaps all previously accepted boxes below threshold."""
        bx, by, bw, bh = box
        candidate = [bx, by, bx + bw, by + bh]
        for prev_b1, _ in accepted:
            _, pl, pt, pr, pb = prev_b1
            if bbox_iou(candidate, [pl, pt, pr, pb]) >= self.iou_threshold:
                return False
        return True

    def _project_to_views(self, box, int_l, int_t, t1, t2, s, batch_idx, n_samples):
        """Map a sampled box (in scaled intersection space) into both view spaces."""
        bx, by, bw, bh = box

        # Project into view-1 coordinate space
        l1 = bx * s["wmin_inv"] * s["sw1"] + (int_l - t1[1].item()) * s["sw1"]
        r1 = l1 + bw * s["wmin_inv"] * s["sw1"]
        t1c = by * s["hmin_inv"] * s["sh1"] + (int_t - t1[0].item()) * s["sh1"]
        b1c = t1c + bh * s["hmin_inv"] * s["sh1"]

        # Project into view-2 coordinate space
        l2 = bx * s["wmin_inv"] * s["sw2"] + (int_l - t2[1].item()) * s["sw2"]
        r2 = l2 + bw * s["wmin_inv"] * s["sw2"]
        t2c = by * s["hmin_inv"] * s["sh2"] + (int_t - t2[0].item()) * s["sh2"]
        b2c = t2c + bh * s["hmin_inv"] * s["sh2"]

        # Horizontal flip correction
        if t1[4]:
            l1, r1 = self.input_size - r1, self.input_size - l1
        if t2[4]:
            l2, r2 = self.input_size - r2, self.input_size - l2

        # Optional box jittering on view-2 (and propagate delta back to view-1)
        if self.box_jittering:
            src2 = [batch_idx + n_samples, l2, t2c, r2, b2c]
            l2, t2c, r2, b2c = jitter_box(l2, t2c, r2, b2c, self.box_jittering_ratio)
            flipped = bool(t1[4]) ^ bool(t2[4])
            if flipped:
                l1 = l1 - (r2 - src2[3]) / s["sw2"] * s["sw1"]
                r1 = r1 - (l2 - src2[1]) / s["sw2"] * s["sw1"]
            else:
                l1 = l1 + (l2 - src2[1]) / s["sw2"] * s["sw1"]
                r1 = r1 + (r2 - src2[3]) / s["sw2"] * s["sw1"]
            t1c = t1c + (t2c - src2[2]) / s["sh2"] * s["sh1"]
            b1c = b1c + (b2c - src2[4]) / s["sh2"] * s["sh1"]

        box1 = [batch_idx,            l1, t1c, r1, b1c]
        box2 = [batch_idx + n_samples, l2, t2c, r2, b2c]
        return box1, box2