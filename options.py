import argparse

parser = argparse.ArgumentParser()

# 训练主参数
parser.add_argument("--epochs", type=int, default=150, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=8, help="每 GPU batch size")
parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
parser.add_argument("--grad_clip", type=float, default=0.5, help="梯度裁剪阈值（梯度范数上限），0 表示不裁剪")
parser.add_argument("--patch_size", type=int, default=128, help="训练 patch 边长")
parser.add_argument("--num_workers", type=int, default=16, help="DataLoader workers")
parser.add_argument("--num_gpus", type=int, default=8, help="使用的 GPU 数")
parser.add_argument("--precision", type=str, default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"], help="混合精度")
parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="梯度累积步数")
parser.add_argument("--pixel_loss_type", type=str, default="l1", choices=["l1", "charbonnier"],
                    help="像素级损失类型：l1 或 charbonnier")
parser.add_argument("--loss_w_l1", type=float, default=1.0, help="像素损失（L1/Charbonnier）权重")
parser.add_argument("--loss_w_cosine", type=float, default=0.5, help="Cosine loss 权重")
parser.add_argument("--loss_w_perceptual", type=float, default=0.01, help="Perceptual loss 权重，0 表示不使用")

# 数据路径（常规多任务）
parser.add_argument("--data_file_dir", type=str, default="data_dir/")
parser.add_argument("--denoise_dir", type=str, default="data/Train/Denoise/")
parser.add_argument("--gopro_dir", type=str, default="data/Train/Deblur/")
parser.add_argument("--enhance_dir", type=str, default="data/Train/Enhance/")
parser.add_argument("--derain_dir", type=str, default="data/Train/Derain/")
parser.add_argument("--dehaze_dir", type=str, default="data/Train/Dehaze/")
parser.add_argument("--de_type", nargs="+", default=["denoise_15", "denoise_25", "denoise_50", "derain", "dehaze", "deblur", "enhance"])

# 输出与日志
parser.add_argument("--ckpt_dir", type=str, default="AdaIR", help="checkpoint 保存目录")
parser.add_argument("--wblogger", type=str, default="AdaIR", help="wandb 项目名，空则不用 wandb")
parser.add_argument("--wandb_offline", action="store_true", help="wandb 离线先存本地")

# Task 7 水下增强：根目录 data/Train/uie/，下含 uieb/ lsui/ euvp/（各带 input/、target/），可选 test_list.txt
parser.add_argument("--uie_data_dir", type=str, default="data/Train/uie/", help="UIE 数据根目录（下含 uieb/lsui/euvp，各含 input/ 与 target/）")
parser.add_argument("--train_uie_only", action="store_true", help="仅用 Task 7 混合数据训练")

# 预训练加载（扩展 5->8 task 时用默认即可）
parser.add_argument("--resume_ckpt", type=str, default=None, help="预训练 .ckpt 路径，加载时默认 5->8 task")

options = parser.parse_args()

# 由 uie_data_dir 派生的路径（与 data/Train/uie/ 实际目录一致：小写 uieb/lsui/euvp）
def _uie_paths():
    base = options.uie_data_dir.rstrip("/") + "/"
    options.uieb_dir = base + "uieb/"
    options.lsui_dir = base + "lsui/"
    options.euvp_dir = base + "euvp/"
    options.uieb_test_list = base + "test_list.txt"  # 可选：列出 90 张 UIEB 测试图文件名，训练时从 UIEB 中排除
_uie_paths()

