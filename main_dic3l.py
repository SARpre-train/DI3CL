#!/usr/bin/env python

# Copyright (c) 2025, Shaanxi Yuanyi Intelligent Technology Co., Ltd.
# This file is part of a project licensed under the MIT License.
# It is developed based on the MoCo project by Meta Platforms, Inc.
# Original MoCo repository: https://github.com/facebookresearch/moco
#
# This project includes significant modifications tailored for SAR land-cover classification,
# including the design of domain-specific modules and the use of large-scale SAR datasets
# to improve performance and generalization on downstream SAR tasks.


import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torch_npu

import dic3l.builder
from dic3l.loader import (
    SAR_dataset, Compose, ColorJitter, Normalize, ToTensor,
    RandomResizedCrop, RandomApply, GaussianBlur,
    RandomHorizontalFlip, TwoCropsTransform, RandomGrayscale,
    decompose_collated_batch,
)
from dic3l.box_generator import BoxGenerator


# ── Constants ─────────────────────────────────────────────────────────────────
INPUT_SIZE = 448
NUM_PATCHES_PER_IMAGE = 10   # must match builder configuration
GLOBAL_LOSS_WEIGHT = 0.8
LOCAL_LOSS_WEIGHT  = 0.2
SIM_LOSS_WEIGHT    = 5.0


# =============================================================================
# CLI Arguments
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
    )
    parser = argparse.ArgumentParser(description="DIC3L SAR Pre-training")

    # Data & Architecture
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet50",
                        choices=model_names)

    # Training Hyperparameters
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("-b", "--batch-size", default=256, type=int)
    parser.add_argument("--lr", "--learning-rate", default=0.03, type=float, dest="lr")
    parser.add_argument("--schedule", default=[120, 160], nargs="*", type=int,
                        help="epochs at which to drop lr by 10x")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float,
                        dest="weight_decay")
    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

    # System / IO
    parser.add_argument("-j", "--workers", default=128, type=int)
    parser.add_argument("-p", "--print-freq", default=1, type=int)
    parser.add_argument("--resume", default="", type=str, metavar="PATH")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--npu", default=None, type=int)

    # Distributed Training
    parser.add_argument("--world-size", default=-1, type=int)
    parser.add_argument("--rank", default=-1, type=int)
    parser.add_argument("--dist-url", default="env://", type=str)
    parser.add_argument("--dist-backend", default="hccl", type=str)
    parser.add_argument("--multiprocessing-distributed", action="store_true")

    # DIC3L / MoCo Specific
    parser.add_argument("--dic3l-dim", default=128, type=int)
    parser.add_argument("--dic3l-k",   default=65536, type=int)
    parser.add_argument("--dic3l-m",   default=0.999, type=float)
    parser.add_argument("--dic3l-t",   default=0.07,  type=float)
    parser.add_argument("--mlp",       action="store_true", help="use MLP projection head")
    parser.add_argument("--aug-plus",  action="store_true", help="use MoCo v2 augmentation")

    return parser


# =============================================================================
# Entry Point
# =============================================================================

def main():
    args = build_parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("Seeded training enables CUDNN determinism, which may slow training.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    nnpus_per_node = torch_npu.npu.device_count()

    if args.multiprocessing_distributed:
        args.world_size = nnpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=nnpus_per_node, args=(nnpus_per_node, args))
    else:
        main_worker(args.npu, nnpus_per_node, args)


# =============================================================================
# Main Worker (per-process entry for distributed training)
# =============================================================================

def main_worker(npu: int, nnpus_per_node: int, args):
    args.npu = npu

    # Suppress output on non-master processes
    if args.multiprocessing_distributed and args.npu != 0:
        builtins.print = lambda *a, **kw: None

    # ── Distributed Init ──────────────────────────────────────────────────────
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * nnpus_per_node + npu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"=> creating model '{args.arch}'")
    model = dic3l.builder.DI3CL(
        models.__dict__[args.arch],
        args.dic3l_dim, args.dic3l_k, args.dic3l_m, args.dic3l_t, args.mlp,
    )

    if args.distributed:
        if npu is not None:
            torch_npu.npu.set_device(npu)
            model.npu(npu)
            args.batch_size = int(args.batch_size / nnpus_per_node)
            args.workers = int((args.workers + nnpus_per_node - 1) / nnpus_per_node)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[npu])
        else:
            model.npu()
            model = nn.parallel.DistributedDataParallel(model)
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # ── Loss & Optimizer ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss().npu(args.npu)
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )

    # ── Resume from Checkpoint ────────────────────────────────────────────────
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            loc = f"npu:{args.npu}" if args.npu is not None else None
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    cudnn.benchmark = True

    # ── Data Loading ──────────────────────────────────────────────────────────
    train_loader, train_sampler = build_dataloader(args)

    # ── Training Loop ─────────────────────────────────────────────────────────
    acc1_list, acc5_list, loss_list = [], [], []
    is_master = not args.multiprocessing_distributed or args.rank % nnpus_per_node == 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        avg_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, args,
            acc1_list, acc5_list,
        )
        loss_list.append(avg_loss)

        if is_master:
            save_checkpoint(
                {"epoch": epoch + 1, "arch": args.arch,
                 "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
                filename=f"checkpoint_{epoch:04d}.pth",
            )

    if is_master:
        np.save("acc1_list.npy", np.array([t.item() for t in acc1_list]))
        np.save("acc5_list.npy", np.array([t.item() for t in acc5_list]))
        np.save("loss_list.npy", np.array(loss_list))


# =============================================================================
# Data Loading
# =============================================================================

def build_dataloader(args):
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.aug_plus:
        # MoCo v2 / SimCLR augmentation
        augmentation = [
            RandomResizedCrop(INPUT_SIZE, scale=(0.2, 1.0)),
            RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    else:
        # MoCo v1 / InstDisc augmentation
        print("Using MoCo v1 augmentation")
        augmentation = [
            RandomResizedCrop(INPUT_SIZE, scale=(0.2, 1.0)),
            ColorJitter(0.4, 0.4, 0, 0),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]

    train_dataset = SAR_dataset(
        image_dir=args.data,
        transform=TwoCropsTransform(Compose(augmentation, with_trans_info=True)),
    )
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if args.distributed else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    return train_loader, train_sampler


# =============================================================================
# Training — Single Epoch
# =============================================================================

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args,
                    acc1_list, acc5_list) -> float:
    box_generator = BoxGenerator(
        input_size=INPUT_SIZE,
        min_size=32,
        num_patches_per_image=NUM_PATCHES_PER_IMAGE,
        box_jittering=False,
        box_jittering_ratio=0,
        iou_threshold=0.5,
        grid_based_box_gen=True,
    )

    # Meters
    batch_time  = AverageMeter("Time",        ":6.3f")
    data_time   = AverageMeter("Data",        ":6.3f")
    sim_loss_m  = AverageMeter("Sim_Loss",    ":.4e")
    global_loss_m = AverageMeter("Global_Loss", ":.4e")
    local_loss_m  = AverageMeter("Local_Loss",  ":.4e")
    total_loss_m  = AverageMeter("Total_Loss",  ":.4e")
    g_top1 = AverageMeter("Global_Acc@1", ":6.2f")
    g_top5 = AverageMeter("Global_Acc@5", ":6.2f")
    l_top1 = AverageMeter("Local_Acc@1",  ":6.2f")
    l_top5 = AverageMeter("Local_Acc@5",  ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, sim_loss_m, global_loss_m,
         local_loss_m, total_loss_m, g_top1, g_top5, l_top1, l_top5],
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()
    end = time.time()

    for i, views in enumerate(train_loader):
        data_time.update(time.time() - end)

        # ── Prepare Inputs ────────────────────────────────────────────────────
        images, transf, _, _ = decompose_collated_batch(views)
        boxes = box_generator.generate(transf)

        images[0] = images[0].npu(args.npu, non_blocking=True)
        images[1] = images[1].npu(args.npu, non_blocking=True)
        box1 = boxes[0].npu(args.npu, non_blocking=True)
        box2 = boxes[1].npu(args.npu, non_blocking=True)
        box2[:, 0] -= images[0].shape[0]   # offset indices by batch size

        # ── Forward ───────────────────────────────────────────────────────────
        output, target, output2, target2, similarity_loss = model(
            im_q=images[0], im_k=images[1], box1=box1, box2=box2,
        )

        global_loss = criterion(output,  target)  * GLOBAL_LOSS_WEIGHT
        local_loss  = criterion(output2, target2) * LOCAL_LOSS_WEIGHT
        sim_loss    = similarity_loss              * SIM_LOSS_WEIGHT
        total_loss  = global_loss + local_loss + sim_loss

        # ── Metrics ───────────────────────────────────────────────────────────
        g_acc1, g_acc5 = accuracy(output,  target,  topk=(1, 5))
        l_acc1, l_acc5 = accuracy(output2, target2, topk=(1, 5))
        acc1_list.append(g_acc1[0])
        acc5_list.append(g_acc5[0])

        n = images[0].size(0)
        sim_loss_m.update(sim_loss.item(), n)
        global_loss_m.update(global_loss.item(), n)
        local_loss_m.update(local_loss.item(), n)
        total_loss_m.update(total_loss.item(), n)
        g_top1.update(g_acc1[0], n);  g_top5.update(g_acc5[0], n)
        l_top1.update(l_acc1[0], n);  l_top5.update(l_acc5[0], n)

        # ── Backward ──────────────────────────────────────────────────────────
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return total_loss_m.avg


# =============================================================================
# Utilities
# =============================================================================

def adjust_learning_rate(optimizer, epoch: int, args):
    lr = args.lr
    if args.cos:
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(state: dict, filename: str = "checkpoint.pth", is_best: bool = False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth")


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Return top-k accuracy (%) for each k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [
            correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)
            for k in topk
        ]


class AverageMeter:
    """Tracks running average and current value."""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name, self.fmt = name, fmt
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return ("{name} {val" + self.fmt + "} ({avg" + self.fmt + "})").format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        width = len(str(num_batches))
        self.batch_fmtstr = "[{:" + str(width) + "d}/" + f"{num_batches:{width}d}]"
        self.meters, self.prefix = meters, prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(m) for m in self.meters]
        print("\t".join(entries))


# =============================================================================

if __name__ == "__main__":
    main()
