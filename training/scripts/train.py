"""Train the shadow-detection model on the full annotated dataset.

- Trains on all annotated samples (no validation holdout)
- Fixed-epoch cosine LR schedule, no early stopping
- ResNet-50 + 19 hand-crafted geometric features
- BatchNorm in shared projection; AMP enabled on CUDA
- Horizontal-flip augmentation with side+direction label mirroring

Usage:
    python scripts/train.py
    python scripts/train.py training.num_epochs=2 run_name=smoke   # quick smoke test
"""

from __future__ import annotations

from shadow_detection.gpu import select_free_gpu

select_free_gpu()

import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch import nn, optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from shadow_detection.data import (  # noqa: E402
    compute_target_stats,
    load_annotations_with_features,
)
from shadow_detection.dataset import ShadowDataset  # noqa: E402
from shadow_detection.model import ShadowModel  # noqa: E402
from shadow_detection.runtime import (  # noqa: E402
    dataloader_kwargs,
    get_device,
    print_runtime_summary,
    use_amp,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = get_device()
    print_runtime_summary()

    seed = int(cfg.training.seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    save_dir = Path(cfg.paths.output_dir) / cfg.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading all annotations and extracting geo features...")
    all_samples = load_annotations_with_features(Path(cfg.paths.train_dir), cfg.data.img_w)
    print(f"Total samples: {len(all_samples)}")

    target_stats = compute_target_stats(all_samples)
    for k, v in target_stats.items():
        print(f"  {k:>20s}: mean={v['mean']:.2f}, std={v['std']:.2f}")

    # Save normalization stats so inference can denormalize regression outputs
    torch.save(target_stats, save_dir / "target_stats.pth")

    ds = ShadowDataset(
        all_samples,
        target_stats,
        input_size=tuple(cfg.data.input_size),
        channel_means=list(cfg.data.channel_means),
        channel_stds=list(cfg.data.channel_stds),
        augment=True,
        flip_prob=cfg.data.flip_prob,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs(cfg.training.num_workers),
    )

    model = ShadowModel(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
        num_geo_features=cfg.model.num_geo_features,
    ).to(device)

    backbone_p = [p for n, p in model.named_parameters() if "backbone" in n]
    head_p = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer = optim.AdamW(
        [
            {"params": backbone_p, "lr": cfg.training.backbone_lr},
            {"params": head_p, "lr": cfg.training.learning_rate},
        ],
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.num_epochs, eta_min=cfg.training.cosine_eta_min
    )
    amp_enabled = use_amp(cfg.training.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    ce = nn.CrossEntropyLoss()
    l1 = nn.SmoothL1Loss()
    w_side = cfg.training.loss_weight_side
    w_reg = cfg.training.loss_weight_regression
    w_dir = cfg.training.loss_weight_direction

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nModel: {n_params/1e6:.1f}M params | "
        f"batches/epoch: {len(loader)} | "
        f"effective batch: {cfg.training.batch_size}"
    )
    print(f"Training on {len(all_samples)} samples for {cfg.training.num_epochs} epochs\n")

    t0 = time.time()
    for epoch in range(cfg.training.num_epochs):
        t_epoch = time.time()
        model.train()
        # Per-task loss accumulators + accuracy counters
        loss_total, loss_side_sum, loss_reg_sum, loss_dir_sum = 0.0, 0.0, 0.0, 0.0
        side_correct, dir_correct, n_seen = 0, 0, 0

        for imgs, sides, regs, dirs, geos in loader:
            imgs = imgs.to(device, non_blocking=True)
            sides = sides.to(device, non_blocking=True)
            regs = regs.to(device, non_blocking=True)
            dirs = dirs.to(device, non_blocking=True)
            geos = geos.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    sl, rp, dl = model(imgs, geos)
                    l_side = ce(sl, sides)
                    l_reg = l1(rp, regs)
                    l_dir = ce(dl, dirs)
                    loss = w_side * l_side + w_reg * l_reg + w_dir * l_dir
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sl, rp, dl = model(imgs, geos)
                l_side = ce(sl, sides)
                l_reg = l1(rp, regs)
                l_dir = ce(dl, dirs)
                loss = w_side * l_side + w_reg * l_reg + w_dir * l_dir
                loss.backward()
                optimizer.step()

            bs = imgs.size(0)
            loss_total += loss.item() * bs
            loss_side_sum += l_side.item() * bs
            loss_reg_sum += l_reg.item() * bs
            loss_dir_sum += l_dir.item() * bs
            side_correct += (sl.argmax(1) == sides).sum().item()
            dir_correct += (dl.argmax(1) == dirs).sum().item()
            n_seen += bs

        scheduler.step()

        epoch_time = time.time() - t_epoch
        elapsed = time.time() - t0
        eta = (cfg.training.num_epochs - epoch - 1) * epoch_time
        lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr

        print(
            f"  Ep {epoch+1:3d}/{cfg.training.num_epochs} | "
            f"loss={loss_total/n_seen:.4f} "
            f"(side={loss_side_sum/n_seen:.3f} reg={loss_reg_sum/n_seen:.3f} "
            f"dir={loss_dir_sum/n_seen:.3f}) | "
            f"side_acc={side_correct/n_seen:.3f} dir_acc={dir_correct/n_seen:.3f} | "
            f"lr=({lr:.2e}, {head_lr:.2e}) | "
            f"{epoch_time:.0f}s | elapsed {elapsed/60:.1f}m | eta {eta/60:.1f}m"
        )

    model_path = save_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved: {model_path}")
    print(f"Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
