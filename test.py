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

    def forward(self, x):
        return self.net(x)


class UIETestDataset(Dataset):
    """UIE test dataset: input/ and target/ with same filenames."""

    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.ids = []
        for f in os.listdir(input_dir):
            if f.startswith(".") or f.startswith("__MACOSX") or os.path.isdir(os.path.join(input_dir, f)):
                continue
            if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            if os.path.isfile(os.path.join(target_dir, f)):
                self.ids.append(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fname = self.ids[idx]
        degrad = np.array(Image.open(os.path.join(self.input_dir, fname)).convert("RGB"))
        clean = np.array(Image.open(os.path.join(self.target_dir, fname)).convert("RGB"))
        degrad = crop_img(degrad, base=16)
        clean = crop_img(clean, base=16)
        degrad = torch.from_numpy(degrad.transpose(2, 0, 1)).float() / 255.0
        clean = torch.from_numpy(clean.transpose(2, 0, 1)).float() / 255.0
        name = os.path.splitext(fname)[0]
        return [name], degrad.unsqueeze(0), clean.unsqueeze(0)


def test_UIE(net, dataset, dataset_name="UIE"):
    psnr = AverageMeter()
    ssim = AverageMeter()
    net.eval()
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=dataset_name):
            [name], degrad, clean = dataset[i]
            degrad = degrad.cuda()
            clean = clean.cuda()
            out = net(degrad)
            restored = out[0] if isinstance(out, (tuple, list)) else out
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
        if not os.path.isdir(inp) or not os.path.isdir(tgt):
            print("{}: skip (missing input/ or target/)".format(name))
            continue
        dataset = UIETestDataset(inp, tgt)
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
