import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import AdaIRTrainDataset, get_uie_train_val_datasets_and_sampler
from net.model import AdaIR, DEBUG_MODE
from utils.checkpoint_utils import load_adair_ckpt_for_uie_finetune
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.pytorch_ssim import ssim as pytorch_ssim
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics import MeanMetric


class AdaIRModel(pl.LightningModule):
    def __init__(self, lr=None):
        super().__init__()
        # lr=None：使用动态学习率（双组 + warmup cosine）；lr 为 float：全程固定学习率
        self.lr = lr
        self.net = AdaIR(decoder=True)
        self.loss_fn = nn.L1Loss()
        if DEBUG_MODE:
            with torch.no_grad():
                for m in self.net.modules():
                    if getattr(m, "is_sft", False):
                        C = m.channels
                        dummy_feat = torch.randn(1, C, 16, 16)
                        dummy_depth = torch.full((1, 1, 16, 16), 0.5)
                        out = m(dummy_depth, dummy_feat)
                        err = (out - dummy_feat).abs().max().item()
                        assert err < 1e-6, "SFT zero-init check failed, max diff={}".format(err)
            print("[DEBUG] SFT zero-init check passed (out = feat).")
        # 用于汇总每个 epoch 的均值（DDP 下会跨卡同步）
        self._epoch_loss = MeanMetric(sync_on_compute=True)
        self._epoch_psnr = MeanMetric(sync_on_compute=True)
        self._epoch_ssim = MeanMetric(sync_on_compute=True)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        if len(batch) == 4:
            ([clean_name, de_id], degrad_patch, clean_patch, depth_map) = batch
            restored = self.net(degrad_patch, depth_map)
            # 启动时打印一次：确认 DataLoader 把深度图传进来了，ECCM 会执行
            if self.current_epoch == 0 and batch_idx == 0 and getattr(self, "_depth_usage_logged", True) is True:
                self._depth_usage_logged = False
                print("\n[DataLoader] ✅ 本 run 使用 4 元组 batch，depth_map 已传入网络，ECCM 会参与前向与反传。\n")
        else:
            ([clean_name, de_id], degrad_patch, clean_patch) = batch
            restored = self.net(degrad_patch)
            if self.current_epoch == 0 and batch_idx == 0 and getattr(self, "_depth_usage_logged", True) is True:
                self._depth_usage_logged = False
                print("\n[DataLoader] ⚠️ 本 run 使用 3 元组 batch，depth_map 始终为 None，ECCM 不会执行！若需训练 ECCM，请使用 --train_uie_only 及带 depth 的数据。\n")

        loss = self.loss_fn(restored, clean_patch)

        # 每个 step 的 PSNR、SSIM（不参与反传，仅用于监控）
        with torch.no_grad():
            restored_d = restored.detach().clamp(0, 1)
            clean_d = clean_patch.detach().clamp(0, 1)
            mse = F.mse_loss(restored_d, clean_d).clamp(min=1e-10)
            step_psnr = (10 * torch.log10(1.0 / mse)).float()
            step_ssim = pytorch_ssim(restored_d, clean_d).float()

        # 每个 step 记录到 wandb/tensorboard
        self.log("step/train_loss", loss, on_step=True, on_epoch=False)
        self.log("step/train_psnr", step_psnr, on_step=True, on_epoch=False)
        self.log("step/train_ssim", step_ssim, on_step=True, on_epoch=False)

        # 累积用于 epoch 均值
        self._epoch_loss.update(loss)
        self._epoch_psnr.update(step_psnr)
        self._epoch_ssim.update(step_ssim)

        return loss

    def on_after_backward(self):
        if not DEBUG_MODE:
            return
        with torch.no_grad():
            grads = []
            for m in self.net.modules():
                if getattr(m, "is_sft", False):
                    if m.scale_conv2.weight.grad is not None:
                        grads.append(m.scale_conv2.weight.grad.abs().mean().item())
                    if m.shift_conv2.weight.grad is not None:
                        grads.append(m.shift_conv2.weight.grad.abs().mean().item())
            if grads:
                g_mean = float(np.mean(grads))
                self.log("step/sft_grad_mean", g_mean, on_step=True, on_epoch=False)

    def on_train_epoch_end(self):
        # 每个 epoch 结束记录本 epoch 平均 loss / PSNR / SSIM（含下划线 key 供 best_ckpt 文件名格式化）
        psnr_val = self._epoch_psnr.compute()
        ssim_val = self._epoch_ssim.compute()
        self.log("epoch/train_loss", self._epoch_loss.compute())
        self.log("epoch/train_psnr", psnr_val)
        self.log("epoch/train_ssim", ssim_val)
        self.log("epoch_train_psnr", psnr_val)
        self.log("epoch_train_ssim", ssim_val)
        self._epoch_loss.reset()
        self._epoch_psnr.reset()
        self._epoch_ssim.reset()
    
    def lr_scheduler_step(self, scheduler, metric):
        # 按 epoch 步进；不传 epoch 避免弃用警告，由 scheduler 内部计数
        scheduler.step()

    def validation_step(self, batch, batch_idx):
        """验证一步：与 training_step 一致的前向与 loss，用于 UIE 等有 val_loader 的场景。"""
        if len(batch) == 4:
            ([_, _], degrad_patch, clean_patch, depth_map) = batch
            restored = self.net(degrad_patch, depth_map)
        else:
            ([_, _], degrad_patch, clean_patch) = batch
            restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        eccm_params = []
        base_params = []
        for name, param in self.named_parameters():
            if "eccm" in name.lower():
                eccm_params.append(param)
            else:
                base_params.append(param)

        print("\n[Debug] === 优化器参数注册自查 ===")
        print(f"👉 捕获到的 ECCM 参数张量数量: {len(eccm_params)}")
        print(f"👉 捕获到的 Base 参数张量数量: {len(base_params)}")
        if len(eccm_params) == 0:
            raise ValueError("🚨 致命错误：优化器没有捕获到任何 ECCM 参数！请检查模型中的变量命名。")
        print("================================\n")

        weight_decay = 1e-4
        if self.lr is not None:
            # 指定了 --lr：全程固定学习率，所有参数同一 lr，不使用 scheduler
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=weight_decay)
            print("[LR] 使用固定学习率: {}".format(self.lr))
            return [optimizer]
        else:
            # 未指定 --lr：动态学习率，base 1e-5 / eccm 1e-3 + warmup cosine
            optimizer = optim.AdamW(
                [
                    {"params": base_params, "lr": 1e-5},
                    {"params": eccm_params, "lr": 1e-3},
                ],
                weight_decay=weight_decay,
            )
            scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=180)
            print("[LR] 使用动态学习率: base 1e-5, eccm 1e-4 + LinearWarmupCosineAnnealingLR")
            return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer, optimizer_idx=None):
        # 梯度探针：检查 ECCM 最后一层梯度是否存活（backward 后、step 前）
        if hasattr(self.net, "eccm") and hasattr(self.net.eccm, "depth_projector"):
            last_layer = self.net.eccm.depth_projector[-1]
            w = last_layer.weight
            if w.grad is None:
                print("🚨 警报：ECCM 最后一层梯度为 None")
            else:
                abs_mean = w.grad.abs().mean().item()
                print("✅ ECCM 梯度存活，Abs Mean: {:.6f}".format(abs_mean))
        # 梯度裁剪，防止梯度假死/爆炸
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.01)


def main():
    print("Options")
    print(opt)
    if getattr(opt, "wblogger", None):
        logger = WandbLogger(
            project=opt.wblogger,
            name="AdaIR-Train",
            offline=opt.wandb_offline,
            save_dir="wandb",
        )
        # 将全部命令行参数记入 wandb.config（首次访问可能触发 init，网络异常时会卡住）
        try:
            _config = logger.experiment.config
            if callable(_config):
                _config = _config()
            if hasattr(_config, "update"):
                _config.update(vars(opt), allow_val_change=True)
        except Exception as e:
            print("[wandb] config.update 失败 (可忽略或使用 --wandb_offline):", e)
        if getattr(opt, "wandb_offline", False):
            print("wandb 离线模式：数据将保存在 ./wandb/ 下，联网后执行: wandb sync ./wandb/offline-run-* 可上传")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    if getattr(opt, "train_uie_only", False):
        train_dataset, val_dataset, train_sampler, uie_stats = get_uie_train_val_datasets_and_sampler(opt, train_ratio=0.9)
        # 仅在 rank 0 打印 UIE 数据集统计，便于校验混合与 4:4:2
        if int(os.environ.get("RANK", 0)) == 0:
            n_train = uie_stats["n_train"]
            n_uieb, n_lsui, n_euvp = uie_stats["n_uieb"], uie_stats["n_lsui"], uie_stats["n_euvp"]
            p_uieb = 100.0 * n_uieb / n_train if n_train else 0
            p_lsui = 100.0 * n_lsui / n_train if n_train else 0
            p_euvp = 100.0 * n_euvp / n_train if n_train else 0
            print("[UIE] train: {} (UIEB: {} [{:.1f}%] | LSUI: {} [{:.1f}%] | EUVP: {} [{:.1f}%]) val: {}".format(
                n_train, n_uieb, p_uieb, n_lsui, p_lsui, n_euvp, p_euvp, uie_stats["n_val"]))
            if getattr(opt, "num_gpus", 1) > 1:
                print("[UIE] num_gpus>1: 使用 DistributedSampler，batch 内比例非严格 4:4:2；单卡时才会按 4:4:2 采样")
        # 单卡用 WeightedRandomSampler(40:40:20)；多卡不传 sampler，由 Lightning 在 DDP 下自动加 DistributedSampler
        use_sampler = getattr(opt, "num_gpus", 1) == 1
        trainloader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            sampler=train_sampler if use_sampler else None,
            shuffle=False if use_sampler else True,
            drop_last=True,
            num_workers=opt.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers) if len(val_dataset) > 0 else None
    else:
        trainset = AdaIRTrainDataset(opt)
        trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                                 drop_last=True, num_workers=opt.num_workers)
        val_loader = None

    # 1) 每个 epoch 正常保存到 ckpt_dir
    epoch_ckpt_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        filename="epoch={epoch:03d}-step={step}",
        every_n_epochs=1,
        save_top_k=-1,
    )
    # 2) 效果最好的 ckpt 保存到 ckpt_dir/best_ckpt，分别按 PSNR 和 SSIM 各保留 1 个，不新建文件夹
    best_ckpt_dir = os.path.join(opt.ckpt_dir, "best_ckpt")
    best_psnr_callback = ModelCheckpoint(
        dirpath=best_ckpt_dir,
        filename="best_psnr-step={step}-epoch={epoch:03d}-psnr={epoch_train_psnr:.2f}-ssim={epoch_train_ssim:.4f}",
        monitor="epoch_train_psnr",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    best_ssim_callback = ModelCheckpoint(
        dirpath=best_ckpt_dir,
        filename="best_ssim-step={step}-epoch={epoch:03d}-psnr={epoch_train_psnr:.2f}-ssim={epoch_train_ssim:.4f}",
        monitor="epoch_train_ssim",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    model = AdaIRModel(lr=opt.lr)

    if getattr(opt, "resume_ckpt", None):
        info = load_adair_ckpt_for_uie_finetune(
            model, opt.resume_ckpt,
            old_num_tasks=5, new_num_tasks=8, copy_from_task_index=0,
        )
        print("Checkpoint loaded (5->8 task): expanded =", info.get("expanded", []), "missing =", len(info.get("missing", [])))

    fit_kw = {"model": model, "train_dataloaders": trainloader}
    if val_loader is not None:
        fit_kw["val_dataloaders"] = val_loader

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[epoch_ckpt_callback, best_psnr_callback, best_ssim_callback],
        precision=opt.precision,
        accumulate_grad_batches=opt.accumulate_grad_batches,
        gradient_clip_val=opt.grad_clip if getattr(opt, "grad_clip", 0.5) > 0 else None,
    )
    trainer.fit(**fit_kw)


if __name__ == '__main__':
    main()