import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import AdaIRTrainDataset, get_uie_train_val_datasets_and_sampler
from net.model import AdaIR
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
from torchvision.utils import make_grid


class AdaIRModel(pl.LightningModule):
    def __init__(self, lr=2e-4):
        super().__init__()
        self.lr = lr
        self.net = AdaIR(decoder=True)
        self.loss_fn = nn.L1Loss()
        # 用于汇总每个 epoch 的均值（DDP 下会跨卡同步）
        self._epoch_loss = MeanMetric(sync_on_compute=True)
        self._epoch_psnr = MeanMetric(sync_on_compute=True)
        self._epoch_ssim = MeanMetric(sync_on_compute=True)

    def forward(self, x):
        I_restored, I_low_res, _ = self.net(x)
        return I_restored, I_low_res

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        I_restored, I_low_res, latent = self.net(degrad_patch)

        # ---------- 非对称监督：辅助色彩 Loss + 最终结构 Loss ----------
        # GT 下采样到与 I_low_res 相同分辨率，与 Encoder 辅助输出对齐
        GT_low_res = F.interpolate(
            clean_patch,
            size=(I_low_res.size(-2), I_low_res.size(-1)),
            mode="bilinear",
            align_corners=False,
        )
        # 辅助色彩 Loss：L1 + Cosine Similarity Loss
        loss_color_l1 = self.loss_fn(I_low_res, GT_low_res)
        B = I_low_res.size(0)
        I_flat = I_low_res.view(B, -1)
        GT_flat = GT_low_res.view(B, -1)
        cos_sim = F.cosine_similarity(I_flat, GT_flat, dim=1)
        loss_color_cos = (1 - cos_sim).mean()
        loss_color = loss_color_l1 + loss_color_cos
        # 最终结构 Loss：原分辨率 L1
        loss_spatial = self.loss_fn(I_restored, clean_patch)
        loss = loss_spatial + 0.2 * loss_color

        # ---------- Instruction 多样性监控：Bottleneck 潜变量方差，趋近 0 表示模式崩溃 ----------
        # 使用 unbiased=False 避免 B=1 时 std 返回 NaN
        with torch.no_grad():
            instruction_std = latent.reshape(B, -1).std(dim=1, unbiased=False).mean().item()
        self.log("step/instruction_std", instruction_std, on_step=True, on_epoch=False)
        self.log("step/loss_spatial", loss_spatial, on_step=True, on_epoch=False)
        self.log("step/loss_color", loss_color, on_step=True, on_epoch=False)

        # 每个 step 的 PSNR、SSIM（不参与反传，仅用于监控）
        with torch.no_grad():
            restored_d = I_restored.detach().clamp(0, 1)
            clean_d = clean_patch.detach().clamp(0, 1)
            mse = F.mse_loss(restored_d, clean_d).clamp(min=1e-10)
            step_psnr = (10 * torch.log10(1.0 / mse)).float()
            step_ssim = pytorch_ssim(restored_d, clean_d).float()

        # 每个 step 记录到 wandb/tensorboard
        self.log("step/train_loss", loss, on_step=True, on_epoch=False)
        self.log("step/train_psnr", step_psnr, on_step=True, on_epoch=False)
        self.log("step/train_ssim", step_ssim, on_step=True, on_epoch=False)

        # ---------- 每隔 N 步保存 I_low_res / GT_low_res 网格图到 Wandb（仅 rank0 避免 DDP 重复） ----------
        if (
            getattr(opt, "wblogger", None)
            and (self.global_step + 1) % 500 == 0
            and batch_idx == 0
            and getattr(self.trainer, "global_rank", 0) == 0
        ):
            with torch.no_grad():
                grid_pred = make_grid(I_low_res.detach().clamp(0, 1), nrow=4, padding=2)
                grid_gt = make_grid(GT_low_res.detach().clamp(0, 1), nrow=4, padding=2)
            logger = self.logger
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                logger.experiment.log(
                    {
                        "train/I_low_res": wandb.Image(grid_pred.permute(1, 2, 0).cpu().numpy()),
                        "train/GT_low_res": wandb.Image(grid_gt.permute(1, 2, 0).cpu().numpy()),
                    },
                    step=self.global_step,
                )

        # 累积用于 epoch 均值
        self._epoch_loss.update(loss)
        self._epoch_psnr.update(step_psnr)
        self._epoch_ssim.update(step_ssim)

        return loss

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
        """Validation 时记录 I_low_res / GT_low_res 网格图，便于肉眼确认 Encoder 修色效果。"""
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        I_restored, I_low_res, _ = self.net(degrad_patch)
        GT_low_res = F.interpolate(
            clean_patch,
            size=(I_low_res.size(-2), I_low_res.size(-1)),
            mode="bilinear",
            align_corners=False,
        )
        # 仅 rank0、第一个 batch 上传图像，避免 DDP 重复与日志过大
        if (
            batch_idx == 0
            and getattr(opt, "wblogger", None)
            and getattr(self.trainer, "global_rank", 0) == 0
        ):
            with torch.no_grad():
                grid_pred = make_grid(I_low_res.detach().clamp(0, 1), nrow=4, padding=2)
                grid_gt = make_grid(GT_low_res.detach().clamp(0, 1), nrow=4, padding=2)
            logger = self.logger
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                logger.experiment.log(
                    {
                        "val/I_low_res": wandb.Image(grid_pred.permute(1, 2, 0).cpu().numpy()),
                        "val/GT_low_res": wandb.Image(grid_gt.permute(1, 2, 0).cpu().numpy()),
                    },
                    step=self.global_step,
                )
        return None

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
        # 项目名中 = 会与 wandb 冲突，统一换成 _
        project = str(opt.wblogger).replace("=", "_")
        # 默认离线，避免内网/代理下 wandb.init() 一直重试卡住；需实时上传时加 --wandb_online
        use_offline = not getattr(opt, "wandb_online", False)
        logger = WandbLogger(
            project=project,
            name="AdaIR-Train",
            offline=use_offline,
            save_dir="wandb",
        )
        _config = logger.experiment.config
        if callable(_config):
            _config = _config()
        if hasattr(_config, "update"):
            _config.update(vars(opt), allow_val_change=True)
        if use_offline:
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