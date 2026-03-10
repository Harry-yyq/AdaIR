import os
import warnings
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
from torchvision.utils import save_image


class AdaIRModel(pl.LightningModule):
    def __init__(self, lr=2e-4):
        super().__init__()
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
            # DAL 零初始化验证：仅在第一个 batch (Epoch 0, Step 0) 执行一次
            if self.current_epoch == 0 and batch_idx == 0 and hasattr(self.net, "dal"):
                with torch.no_grad():
                    raw_depth = depth_map.detach()
                    calibrated_depth = self.net.dal(raw_depth)
                    diff_max = torch.abs(calibrated_depth - raw_depth).max().item()
                    print("[DAL] Epoch 0 Step 0 零初始化检查: |calibrated - raw|.max() = {:.6f}".format(diff_max))
                    if diff_max > 1e-6:
                        warnings.warn(
                            "DAL 零初始化异常：校准前后差异 |calibrated - raw|.max() = {:.6f}，期望接近 0。请检查 DAL 最后一层是否零初始化。".format(diff_max),
                            UserWarning,
                        )
            # 注意：传入模型的 depth_map 保持原 tensor（不 detach），以便梯度回传到 DAL
            restored = self.net(degrad_patch, depth_map)
        else:
            ([clean_name, de_id], degrad_patch, clean_patch) = batch
            restored = self.net(degrad_patch)

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
        # DAL 梯度监控：第一个 Step 的 backward 后，打印 DAL 最后一层 conv3 的梯度均值，确保梯度能传回
        if self.global_step == 0 and hasattr(self.net, "dal") and hasattr(self.net.dal, "conv3"):
            w = self.net.dal.conv3.weight
            if w.grad is not None:
                dal_grad_mean = w.grad.abs().mean().item()
                print("[DAL] Epoch 0 Step 0 backward 后: dal.conv3.weight.grad 平均绝对值 = {:.6f}".format(dal_grad_mean))
                self.log("step/dal_conv3_grad_mean", dal_grad_mean, on_step=True, on_epoch=False)
            else:
                print("[DAL] Epoch 0 Step 0 backward 后: dal.conv3.weight.grad 为 None（该 step 可能未使用深度）")
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

    def validation_step(self, batch, batch_idx):
        """验证一步：计算 val loss，并在第一个 batch 保存重建图与 raw/calibrated 深度对比图。"""
        if len(batch) == 4:
            ([clean_name, de_id], degrad_patch, clean_patch, depth_map) = batch
            raw_depth = depth_map.detach()
            with torch.no_grad():
                calibrated_depth = self.net.dal(raw_depth) if hasattr(self.net, "dal") else raw_depth
            # 验证时同样传入未 detach 的 depth_map，保证与训练一致
            restored = self.net(degrad_patch, depth_map)
        else:
            ([clean_name, de_id], degrad_patch, clean_patch) = batch
            raw_depth = calibrated_depth = None
            restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        with torch.no_grad():
            restored_d = restored.detach().clamp(0, 1)
            clean_d = clean_patch.detach().clamp(0, 1)
            mse = F.mse_loss(restored_d, clean_d).clamp(min=1e-10)
            val_psnr = (10 * torch.log10(1.0 / mse)).float().item()
            val_ssim = pytorch_ssim(restored_d, clean_d).float().item()
        self.log("val/loss", loss)
        self.log("val/psnr", val_psnr)
        self.log("val/ssim", val_ssim)

        # 可视化：仅第一个 validation batch 保存重建图与 raw vs calibrated 深度并排图
        if batch_idx != 0 or raw_depth is None:
            return loss
        log_dir = getattr(self.logger, "log_dir", None) or getattr(self.trainer, "log_dir", None) or "."
        val_vis_dir = os.path.join(log_dir, "val_vis")
        os.makedirs(val_vis_dir, exist_ok=True)
        prefix = "epoch{:03d}".format(self.current_epoch)
        # 重建图：取前 4 张拼成 2x2
        save_image(restored_d[:4].clamp(0, 1), os.path.join(val_vis_dir, "{}_restored.png".format(prefix)), nrow=2)
        # 原始深度 vs 校准深度并排（单通道复制为 3 通道便于查看）：左 raw，右 calibrated
        raw_3 = raw_depth[:1].repeat(1, 3, 1, 1).clamp(0, 1)
        cal_3 = calibrated_depth[:1].repeat(1, 3, 1, 1).clamp(0, 1)
        side_by_side = torch.cat([raw_3, cal_3], dim=3)
        save_image(side_by_side, os.path.join(val_vis_dir, "{}_raw_vs_calibrated_depth.png".format(prefix)), nrow=1)
        return loss

    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=180)

        return [optimizer], [scheduler]


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
        # 将全部命令行参数记入 wandb.config（部分环境里 config 为方法，需先取再 update）
        _config = logger.experiment.config
        if callable(_config):
            _config = _config()
        if hasattr(_config, "update"):
            _config.update(vars(opt), allow_val_change=True)
        if opt.wandb_offline:
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
    model = AdaIRModel(lr=getattr(opt, "lr", 2e-4))

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