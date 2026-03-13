import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import lightning.pytorch as pl

from net.model import AdaIR
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_utils import crop_img


class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)

    def forward(self, x, depth_map=None):
        """图像进、图像出。depth_map 可选，若提供则启用 ECCM 与 Fre 深度分支。"""
        return self.net(x, depth_map)


class UIETestDataset(Dataset):
    """UIE test dataset: input/ 与 target/ 同文件名。可选 depth_dir，与训练约定一致：depth/xxx.npy 或 .png/.jpg/.jpeg，0~1。"""

    def __init__(self, input_dir, target_dir, depth_dir=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.depth_dir = depth_dir  # 若为 None，则纯黑盒，不加载深度
        self.ids = []
        for f in os.listdir(input_dir):
            if f.startswith(".") or f.startswith("__MACOSX") or os.path.isdir(os.path.join(input_dir, f)):
                continue
            if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            if os.path.isfile(os.path.join(target_dir, f)):
                self.ids.append(f)

    def _load_depth(self, fname, H, W):
        """与 dataset_utils 一致：depth/ 下 base.npy 或 base.png/.jpg/.jpeg，归一化到 0~1。"""
        if self.depth_dir is None or not os.path.isdir(self.depth_dir):
            return None
        base_no_ext = os.path.splitext(fname)[0]
        cand_npy = os.path.join(self.depth_dir, base_no_ext + ".npy")
        cand_png = os.path.join(self.depth_dir, base_no_ext + ".png")
        cand_jpg = os.path.join(self.depth_dir, base_no_ext + ".jpg")
        cand_jpeg = os.path.join(self.depth_dir, base_no_ext + ".jpeg")
        depth_arr = None
        if os.path.isfile(cand_npy):
            depth_arr = np.load(cand_npy).astype(np.float32)
        elif os.path.isfile(cand_png):
            depth_arr = np.array(Image.open(cand_png).convert("L"), dtype=np.float32)
        elif os.path.isfile(cand_jpg):
            depth_arr = np.array(Image.open(cand_jpg).convert("L"), dtype=np.float32)
        elif os.path.isfile(cand_jpeg):
            depth_arr = np.array(Image.open(cand_jpeg).convert("L"), dtype=np.float32)
        if depth_arr is None:
            return None
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]
        depth_img = Image.fromarray(depth_arr)
        depth_img = depth_img.resize((W, H), Image.BILINEAR)
        depth_arr = np.array(depth_img).astype(np.float32)
        if depth_arr.max() > 1.0 + 1e-3:
            depth_arr = depth_arr / 255.0
        depth_arr = np.clip(depth_arr, 0.0, 1.0)
        return torch.from_numpy(depth_arr[None, ...]).float()  # (1,H,W)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fname = self.ids[idx]
        degrad = np.array(Image.open(os.path.join(self.input_dir, fname)).convert("RGB"))
        clean = np.array(Image.open(os.path.join(self.target_dir, fname)).convert("RGB"))
        degrad = crop_img(degrad, base=16)
        clean = crop_img(clean, base=16)
        H, W = degrad.shape[0], degrad.shape[1]
        degrad = torch.from_numpy(degrad.transpose(2, 0, 1)).float() / 255.0
        clean = torch.from_numpy(clean.transpose(2, 0, 1)).float() / 255.0
        name = os.path.splitext(fname)[0]
        degrad = degrad.unsqueeze(0)   # (1,3,H,W)
        clean = clean.unsqueeze(0)
        depth = self._load_depth(fname, H, W)
        if depth is not None:
            depth = depth.unsqueeze(0)  # (1,1,H,W)
        return [name], degrad, clean, depth


def test_UIE(net, dataset, dataset_name="UIE"):
    psnr = AverageMeter()
    ssim = AverageMeter()
    net.eval()
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=dataset_name):
            [name], degrad, clean, depth_map = dataset[i]
            degrad = degrad.cuda()
            clean = clean.cuda()
            if depth_map is not None:
                depth_map = depth_map.cuda()
                restored = net(degrad, depth_map)
            else:
                restored = net(degrad)
            restored = torch.clamp(restored, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
    print("{}: PSNR: {:.2f}, SSIM: {:.4f}".format(dataset_name, psnr.avg, ssim.avg))
    return psnr.avg, ssim.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, default="ckpt/adair5d.ckpt", help="checkpoint path")
    parser.add_argument("--uie_test_dir", type=str, default="data/test/uie/",
                        help="UIE test root, expect uieb/lsui/euvp with input/ and target/")
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)

    net = AdaIRModel.load_from_checkpoint(args.ckpt_path).net.cuda()
    net.eval()

    # 相对路径以脚本所在目录为基准，避免 cwd 不同导致找不到数据
    uie_dir = args.uie_test_dir
    if not os.path.isabs(uie_dir):
        _root = os.path.dirname(os.path.abspath(__file__))
        uie_dir = os.path.normpath(os.path.join(_root, uie_dir))
    base = uie_dir.rstrip(os.sep) + os.sep
    results = {}
    for name, sub in [("UIEB", "uieb"), ("LSUI", "lsui"), ("EUVP", "euvp")]:
        inp = os.path.join(base, sub, "input")
        tgt = os.path.join(base, sub, "target")
        depth_dir = os.path.join(base, sub, "depth")
        if not os.path.isdir(inp) or not os.path.isdir(tgt):
            print("{}: skip (missing input/ or target/)".format(name))
            continue
        use_depth = os.path.isdir(depth_dir)
        if use_depth:
            print("{}: 使用 depth/，ECCM + Fre 深度分支参与推理".format(name))
        dataset = UIETestDataset(inp, tgt, depth_dir=depth_dir if use_depth else None)
        if len(dataset) == 0:
            print("{}: skip (no pairs)".format(name))
            continue
        psnr, ssim = test_UIE(net, dataset, dataset_name=name)
        results[name] = (psnr, ssim)

    if results:
        print("-" * 40)
        print("Task 7 UIE Test Summary:")
        for name in ["UIEB", "LSUI", "EUVP"]:
            if name in results:
                p, s = results[name]
                print("  {}: PSNR: {:.2f}, SSIM: {:.4f}".format(name, p, s))


if __name__ == "__main__":
    main()
