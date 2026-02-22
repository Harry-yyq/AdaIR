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


class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)
        self.loss_fn = nn.L1Loss()
        # 用于汇总每个 epoch 的均值（DDP 下会跨卡同步）
        self._epoch_loss = MeanMetric(sync_on_compute=True)
        self._epoch_psnr = MeanMetric(sync_on_compute=True)
        self._epoch_ssim = MeanMetric(sync_on_compute=True)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
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

    def on_train_epoch_end(self):
        # 每个 epoch 结束记录本 epoch 平均 loss / PSNR / SSIM
        self.log("epoch/train_loss", self._epoch_loss.compute())
        self.log("epoch/train_psnr", self._epoch_psnr.compute())
        self.log("epoch/train_ssim", self._epoch_ssim.compute())
        self._epoch_loss.reset()
        self._epoch_psnr.reset()
        self._epoch_ssim.reset()
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=180)

        return [optimizer],[scheduler]


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
        train_dataset, val_dataset, train_sampler = get_uie_train_val_datasets_and_sampler(opt, train_ratio=0.9)
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

    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1)
    model = AdaIRModel()

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
        callbacks=[checkpoint_callback],
        precision=opt.precision,
        accumulate_grad_batches=opt.accumulate_grad_batches,
    )
    trainer.fit(**fit_kw)


if __name__ == '__main__':
    main()