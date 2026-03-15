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

# 验证集定性可视化保存目录
VIS_DIR = "output/visualization"


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


def _tensor_to_uint8_rgb(x):
    """(1,3,H,W) or (1,1,H,W) -> (H,W,3) uint8。单通道会复制为 3 通道便于拼接。"""
    x = x.squeeze(0)
    if x.dim() == 2:
        x = x.unsqueeze(0).expand(3, -1, -1)
    elif x.shape[0] == 1:
        x = x.expand(3, -1, -1)
    x = x.permute(1, 2, 0).cpu().numpy()
    x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    return x


def _scale_to_heatmap_uint8(scale, dim=1):
    """scale (1,C,H,W) -> 通道均值 abs -> 热力图 (H,W,3) uint8。"""
    heat = torch.mean(scale.abs(), dim=dim).squeeze(0).cpu().numpy()
    heat = np.clip(heat, 0.0, None)
    if heat.max() > heat.min() + 1e-6:
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    else:
        heat = np.zeros_like(heat)
    heat = (heat * 255.0).astype(np.uint8)
    try:
        import cv2
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    except Exception:
        heat = np.stack([heat, heat, heat], axis=-1)
    return heat


def test_UIE(net, dataset, dataset_name="UIE"):
    psnr = AverageMeter()
    ssim = AverageMeter()
    net.eval()
    os.makedirs(VIS_DIR, exist_ok=True)
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=dataset_name):
            [name], degrad, clean, _ = dataset[i]
            degrad = degrad.cuda()
            clean = clean.cuda()
            # 每组数据集的第 1 张图：带 return_aux 取深度与 ECCM scale，并保存可视化
            if i == 0:
                out = net(degrad, return_aux=True)
                if isinstance(out, tuple):
                    restored, aux = out
                else:
                    restored = out
                    aux = {}
            else:
                restored = net(degrad)
                aux = {}
            restored = torch.clamp(restored, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            # 第 1 张图：拼接 [原图, 深度图, ECCM_Scale_Heatmap, 恢复图, GT] 并保存
            # eccm_scale 来自 AdaIR 的 aux，即 ECCM.forward 的返回值：scale = torch.tanh(depth_projector(depth_map))，为 tanh 后的调制系数
            if i == 0 and aux:
                depth_map = aux.get("depth_map")
                eccm_scale = aux.get("eccm_scale")  # 已是 depth_projector → tanh 后的 scale，非中间层
                # 原图、恢复图、GT：(1,3,H,W) -> (H,W,3)
                im_input = _tensor_to_uint8_rgb(degrad)
                im_restored = _tensor_to_uint8_rgb(restored)
                im_gt = _tensor_to_uint8_rgb(clean)
                h, w = im_input.shape[0], im_input.shape[1]
                if depth_map is not None:
                    im_depth = _tensor_to_uint8_rgb(depth_map)
                else:
                    im_depth = np.zeros((h, w, 3), dtype=np.uint8)
                if eccm_scale is not None:
                    im_scale = _scale_to_heatmap_uint8(eccm_scale, dim=1)
                else:
                    im_scale = np.zeros((h, w, 3), dtype=np.uint8)
                # 统一缩放到原图尺寸后拼接（depth/scale 可能为 H/8,W/8）
                def _resize_to_hw(arr, target_h, target_w):
                    if arr.shape[0] == target_h and arr.shape[1] == target_w:
                        return arr
                    return np.array(Image.fromarray(arr).resize((target_w, target_h)))
                im_depth = _resize_to_hw(im_depth, h, w)
                im_scale = _resize_to_hw(im_scale, h, w)
                im_restored = _resize_to_hw(im_restored, h, w)
                im_gt = _resize_to_hw(im_gt, h, w)
                row = np.concatenate([im_input, im_depth, im_scale, im_restored, im_gt], axis=1)
                out_path = os.path.join(VIS_DIR, "{}_first_input_depth_scale_restored_gt.png".format(dataset_name))
                Image.fromarray(row).save(out_path)
                print("[Visualization] saved: {}".format(out_path))
    print("{}: PSNR: {:.2f}, SSIM: {:.4f}".format(dataset_name, psnr.avg, ssim.avg))
    return psnr.avg, ssim.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, default="ckpt/adair5d.ckpt",
                        help="AdaIR 训练得到的 Lightning 检查点 (.ckpt)，非 DepthAnythingV2 的 .pth")
    parser.add_argument("--uie_test_dir", type=str, default="data/test/uie/",
                        help="UIE test root, expect uieb/lsui/euvp with input/ and target/")
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)

    # --ckpt_path 必须是「训练得到的 AdaIR Lightning 检查点」(.ckpt)，不是 DepthAnythingV2 的 .pth
    # strict=False：旧 ckpt 可能不含 net.depth_estimator，缺失的键保留为 __init__ 中已加载的 DepthAnythingV2 权重
    try:
        net = AdaIRModel.load_from_checkpoint(args.ckpt_path, strict=False).net.cuda()
    except KeyError as e:
        if "pytorch-lightning_version" in str(e) or "state_dict" in str(e):
            print("错误: 您传入的不是 Lightning 检查点，无法用 load_from_checkpoint 加载。")
            print("  --ckpt_path 应指向「训练保存的 AdaIR 模型」例如: ckpt/best_ckpt/best_psnr-xxx.ckpt")
            print("  不要传入 DepthAnythingV2 的权重路径（如 depth_anything_v2_vits.pth），该权重在 AdaIR 初始化时已自动从 ckpt/DepthAnythingv2/ 加载。")
        raise
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
