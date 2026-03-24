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
import torch.nn.functional as F
from torch_npu.contrib.module import ROIAlign

from dic3l.net import CustomResNet, TwoLayerLinearHead


# =============================================================================
# Loss
# =============================================================================

def _cosine_regression_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Normalized MSE loss equivalent to 2 - 2·cos(preds, targets), averaged over batch."""
    preds   = F.normalize(preds,   dim=1)
    targets = F.normalize(targets, dim=1)
    return 2 - 2 * (preds * targets).sum() / preds.size(0)


# =============================================================================
# DI3CL Model
# =============================================================================

class DI3CL(nn.Module):
    """DI3CL.

    Maintains a query encoder and an EMA key encoder, plus two FIFO queues
    (global and low-level) for negative sample storage.

    Args:
        base_encoder:  torchvision ResNet constructor
        dic3l_dim:     projection embedding dimension (default: 128)
        dic3l_k:       queue size / number of negatives (default: 65536)
        dic3l_m:       EMA momentum for key encoder update (default: 0.999)
        dic3l_t:       softmax temperature (default: 0.07)
        mlp:           must be True; only MLP-head variant is supported
    """

    def __init__(self, base_encoder, dic3l_dim: int = 128, dic3l_k: int = 65536,
                 dic3l_m: float = 0.999, dic3l_t: float = 0.07, mlp: bool = True):
        super().__init__()

        if not mlp:
            raise ValueError("Only the MLP-head variant (mlp=True) is supported.")

        self.dic3l_k = dic3l_k
        self.dic3l_m = dic3l_m
        self.dic3l_t = dic3l_t

        # ── Encoders ──────────────────────────────────────────────────────────
        self.encoder_q = CustomResNet(base_encoder, dim=dic3l_dim)
        self.encoder_k = CustomResNet(base_encoder, dim=dic3l_dim)
        _copy_params_and_freeze(self.encoder_q, self.encoder_k)

        # ── Local projectors (ROI-pooled features → embedding) ────────────────
        self.local_proj_q = TwoLayerLinearHead(2048, 2048, dic3l_dim, batch_norm=True)
        self.local_proj_k = TwoLayerLinearHead(2048, 2048, dic3l_dim, batch_norm=True)
        _copy_params_and_freeze(self.local_proj_q, self.local_proj_k)

        # ── Local predictor (query-side only, BYOL-style) ─────────────────────
        self.local_predictor = TwoLayerLinearHead(dic3l_dim, 2048, dic3l_dim, batch_norm=True)

        # ── ROI Align (spatial_scale = 1/32 for stride-32 backbone) ──────────
        self.roi_align = ROIAlign(
            output_size=(1, 1), sampling_ratio=0,
            spatial_scale=0.03125, aligned=True,
        )

        # ── Queues ────────────────────────────────────────────────────────────
        self.register_buffer("global_queue", F.normalize(torch.randn(dic3l_dim, dic3l_k), dim=0))
        self.register_buffer("local_queue",  F.normalize(torch.randn(dic3l_dim, dic3l_k), dim=0))
        self.register_buffer("queue_ptr",    torch.zeros(1, dtype=torch.long))

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, im_q, im_k, box1, box2):
        """
        Args:
            im_q: query image batch   (N, C, H, W)
            im_k: key image batch     (N, C, H, W)
            box1: ROI boxes for view1 (M, 5) — [batch_idx, l, t, r, b]
            box2: ROI boxes for view2 (M, 5) — [batch_idx+N, l, t, r, b]

        Returns:
            global_logits, global_labels,  # for global contrastive loss
            local_logits,  local_labels,   # for low-level contrastive loss
            region_loss                    # local region regression loss
        """
        # ── Query forward ─────────────────────────────────────────────────────
        q_global, q_low, q_features = self.encoder_q(im_q)
        q_global = F.normalize(q_global, dim=1)
        q_low    = F.normalize(q_low,    dim=1)

        q_region = self.roi_align(q_features, box1).squeeze()
        q_region = self.local_proj_q(q_region)

        # ── Key forward (no gradient, with BN shuffle) ────────────────────────
        with torch.no_grad():
            self._momentum_update_key_encoder()

            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k_global, k_low, k_features = self.encoder_k(im_k)
            k_global  = F.normalize(k_global, dim=1)
            k_low     = F.normalize(k_low,    dim=1)
            k_global  = self._batch_unshuffle_ddp(k_global,  idx_unshuffle)
            k_low     = self._batch_unshuffle_ddp(k_low,     idx_unshuffle)
            k_features = self._batch_unshuffle_ddp(k_features, idx_unshuffle)

            k_region = self.roi_align(k_features, box2).squeeze()
            k_region = self.local_proj_k(k_region)

        # ── Global contrastive logits ─────────────────────────────────────────
        global_logits, global_labels = self._contrastive_logits(
            q_global, k_global, self.global_queue
        )

        # ── Low-level contrastive logits ──────────────────────────────────────
        local_logits, local_labels = self._contrastive_logits(
            q_low, k_low, self.local_queue
        )

        # ── Region regression loss ────────────────────────────────────────────
        q_region_pred = self.local_predictor(q_region)
        region_loss   = _cosine_regression_loss(q_region_pred, k_region)

        # ── Queue update ──────────────────────────────────────────────────────
        self._dequeue_and_enqueue(k_global, k_low)

        return global_logits, global_labels, local_logits, local_labels, region_loss

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _contrastive_logits(self, q, k, queue):
        """Build (N, 1+K) logit matrix and zero-index labels for one contrastive head."""
        l_pos  = torch.einsum("nc,nc->n",  [q, k]).unsqueeze(-1)          # (N, 1)
        l_neg  = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])   # (N, K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.dic3l_t
        labels = torch.zeros(logits.size(0), dtype=torch.long).npu()
        return logits, labels

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """EMA update: param_k ← m·param_k + (1−m)·param_q."""
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.dic3l_m + p_q.data * (1.0 - self.dic3l_m)
        for p_q, p_k in zip(self.local_proj_q.parameters(), self.local_proj_k.parameters()):
            p_k.data = p_k.data * self.dic3l_m + p_q.data * (1.0 - self.dic3l_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, global_keys, local_keys):
        """FIFO queue update for both global and local feature banks."""
        global_keys = concat_all_gather(global_keys)
        local_keys  = concat_all_gather(local_keys)

        batch_size = global_keys.size(0)
        ptr = int(self.queue_ptr)
        assert self.dic3l_k % batch_size == 0, "Queue size must be divisible by batch size."

        self.global_queue[:, ptr:ptr + batch_size] = global_keys.T
        self.local_queue[:, ptr:ptr + batch_size]  = local_keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.dic3l_k

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Shuffle samples across GPUs to decorrelate BatchNorm statistics."""
        batch_size_local = x.size(0)
        x_all            = concat_all_gather(x)
        batch_size_all   = x_all.size(0)
        num_npus         = batch_size_all // batch_size_local

        idx_shuffle = torch.randperm(batch_size_all).npu()
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)

        npu_idx  = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_npus, -1)[npu_idx]
        return x_all[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Restore the original sample order after key encoder forward."""
        batch_size_local = x.size(0)
        x_all            = concat_all_gather(x)
        batch_size_all   = x_all.size(0)
        num_npus         = batch_size_all // batch_size_local

        npu_idx  = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_npus, -1)[npu_idx]
        return x_all[idx_this]


# =============================================================================
# Utilities
# =============================================================================

def _copy_params_and_freeze(src: nn.Module, dst: nn.Module):
    """Copy parameters from src to dst and disable gradient on dst."""
    for p_src, p_dst in zip(src.parameters(), dst.parameters()):
        p_dst.data.copy_(p_src.data)
        p_dst.requires_grad = False


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather tensor across all DDP ranks (no gradient)."""
    gathered = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor, async_op=False)
    return torch.cat(gathered, dim=0)
