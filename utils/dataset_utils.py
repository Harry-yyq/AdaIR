import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation


# --------------- Task 7: Underwater Image Enhancement (UIE) ---------------
# Task ID for UIE in AdaIR (7th category)
UIE_TASK_ID = 7

# Target batch ratio for WeightedRandomSampler: UIEB : LSUI : EUVP = 40% : 40% : 20%
UIE_SOURCE_RATIOS = (0.4, 0.4, 0.2)  # (UIEB, LSUI, EUVP)
UIE_SOURCE_UIEB, UIE_SOURCE_LSUI, UIE_SOURCE_EUVP = 0, 1, 2


def load_uieb_test_ids(test_list_path):
    """Read test_list.txt and return a set of basenames to exclude from UIEB train/val.
    File format: one image filename per line (e.g. '890-1.png' or '890-1').
    """
    if not os.path.isfile(test_list_path):
        return set()
    exclude = set()
    with open(test_list_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            # normalize to basename without path
            base = os.path.basename(name)
            exclude.add(base)
            # also add without extension so '890-1' matches '890-1.png'
            base_no_ext = os.path.splitext(base)[0]
            exclude.add(base_no_ext)
    return exclude


def _ensure_min_size(img, patch_size):
    """Pad or resize so that H, W >= patch_size. img: numpy HWC, RGB [0,255]. Returns HWC."""
    h, w = img.shape[0], img.shape[1]
    if h >= patch_size and w >= patch_size:
        return img
    scale = 1.0
    if h < patch_size or w < patch_size:
        scale = max(patch_size / h, patch_size / w)
    new_h = max(int(round(h * scale)), patch_size)
    new_w = max(int(round(w * scale)), patch_size)
    img = np.array(
        Image.fromarray(img.astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR)
    )
    return img


def _uie_augment_pair(deg_patch, clean_patch):
    """Apply same RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(90) to both. NO ColorJitter."""
    # Use 1-7 only: mode 0 in data_augmentation uses .numpy() which fails on ndarray
    mode = random.randint(1, 7)
    from utils.image_utils import data_augmentation
    deg_patch = data_augmentation(deg_patch, mode)
    clean_patch = data_augmentation(clean_patch, mode)
    return deg_patch, clean_patch


class UnderwaterDataset(Dataset):
    """Unified dataset for Task 7 (Underwater Image Enhancement).
    Each sample: (degraded_patch, clean_patch) both RGB [0,1] tensors, task_id=7.
    Returns format compatible with AdaIRTrainDataset: ([clean_name, task_id], degrad_patch, clean_patch).
    """

    def __init__(self, sample_list, patch_size, degraded_subdir="input", ref_subdir="ref"):
        """
        sample_list: list of dicts with keys 'degrad_path', 'ref_path', 'source_id' (0=UIEB, 1=LSUI, 2=EUVP).
        patch_size: int.
        degraded_subdir / ref_subdir: optional subdir names if paths are dataset roots.
        """
        self.sample_list = sample_list
        self.patch_size = patch_size
        self.degraded_subdir = degraded_subdir
        self.ref_subdir = ref_subdir
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        item = self.sample_list[idx]
        degrad_path = item["degrad_path"]
        ref_path = item["ref_path"]

        degrad_img = np.array(Image.open(degrad_path).convert("RGB"))
        clean_img = np.array(Image.open(ref_path).convert("RGB"))

        degrad_img = _ensure_min_size(degrad_img, self.patch_size)
        clean_img = _ensure_min_size(clean_img, self.patch_size)

        H, W = degrad_img.shape[0], degrad_img.shape[1]
        if H <= self.patch_size or W <= self.patch_size:
            # edge case: exactly patch_size
            top, left = 0, 0
        else:
            top = random.randint(0, H - self.patch_size)
            left = random.randint(0, W - self.patch_size)
        degrad_patch = degrad_img[top : top + self.patch_size, left : left + self.patch_size]
        clean_patch = clean_img[top : top + self.patch_size, left : left + self.patch_size]

        degrad_patch, clean_patch = _uie_augment_pair(degrad_patch, clean_patch)
        degrad_patch = np.ascontiguousarray(degrad_patch)
        clean_patch = np.ascontiguousarray(clean_patch)

        degrad_patch = self.to_tensor(degrad_patch)
        clean_patch = self.to_tensor(clean_patch)

        clean_name = os.path.splitext(os.path.basename(ref_path))[0]
        return [clean_name, UIE_TASK_ID], degrad_patch, clean_patch


def _collect_pairs_from_dir(root_dir, degraded_subdir="input", ref_subdir="ref", source_id=0, exclude_basenames=None):
    """Collect (degrad_path, ref_path, source_id) from root_dir/input and root_dir/ref. Same filenames."""
    degrad_dir = os.path.join(root_dir, degraded_subdir)
    ref_dir = os.path.join(root_dir, ref_subdir)
    if not os.path.isdir(degrad_dir) or not os.path.isdir(ref_dir):
        return []
    pairs = []
    for fname in os.listdir(degrad_dir):
        if fname.startswith(".") or fname.startswith("__MACOSX") or os.path.isdir(os.path.join(degrad_dir, fname)):
            continue
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        base = os.path.basename(fname)
        if exclude_basenames and (base in exclude_basenames or os.path.splitext(base)[0] in exclude_basenames):
            continue
        ref_path = os.path.join(ref_dir, fname)
        if not os.path.isfile(ref_path):
            continue
        degrad_path = os.path.join(degrad_dir, fname)
        pairs.append({"degrad_path": degrad_path, "ref_path": ref_path, "source_id": source_id})
    return pairs


def build_uie_mixed_pool_and_split(args, train_ratio=0.9, seed=None):
    """Build mixed UIE pool (excluding UIEB test 90), split into train/val.
    Returns:
        train_samples: list of sample dicts for training
        val_samples: list of sample dicts for validation
        train_source_ids: list of source_id per train sample (for weighted sampler)
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    exclude = load_uieb_test_ids(args.uieb_test_list)
    # 实际目录为 input/ 与 target/（与 data/Train/uie/ 一致）
    uieb = _collect_pairs_from_dir(
        args.uieb_dir, "input", "target", UIE_SOURCE_UIEB, exclude_basenames=exclude
    )
    lsui = _collect_pairs_from_dir(args.lsui_dir, "input", "target", UIE_SOURCE_LSUI)
    euvp = _collect_pairs_from_dir(args.euvp_dir, "input", "target", UIE_SOURCE_EUVP)

    if len(uieb) == 0 and os.path.isdir(args.uieb_dir):
        uieb = _collect_pairs_from_dir(args.uieb_dir, "input", "ref", UIE_SOURCE_UIEB, exclude_basenames=exclude)
    if len(lsui) == 0 and os.path.isdir(args.lsui_dir):
        lsui = _collect_pairs_from_dir(args.lsui_dir, "input", "ref", UIE_SOURCE_LSUI)
    if len(euvp) == 0 and os.path.isdir(args.euvp_dir):
        euvp = _collect_pairs_from_dir(args.euvp_dir, "input", "ref", UIE_SOURCE_EUVP)

    all_samples = uieb + lsui + euvp
    if len(all_samples) == 0:
        raise RuntimeError(
            "UIE mixed pool is empty. Check uie_data_dir and that uieb/lsui/euvp each have input/ and target/ with matching filenames."
        )
    rng.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * train_ratio)
    if n_train < 1:
        n_train = 1
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]
    train_source_ids = [s["source_id"] for s in train_samples]

    return train_samples, val_samples, train_source_ids


def build_uie_weighted_sampler(train_source_ids, num_samples=None, replacement=True):
    """WeightedRandomSampler so that batch ratio is UIEB:LSUI:EUVP = 40:40:20."""
    if num_samples is None:
        num_samples = len(train_source_ids)
    if num_samples == 0:
        raise RuntimeError("Cannot build weighted sampler for empty train set.")
    n_per_source = [0, 0, 0]
    for sid in train_source_ids:
        n_per_source[sid] += 1
    # weight[i] = target_ratio[source_i] / n_source
    target_ratios = list(UIE_SOURCE_RATIOS)
    weights = []
    for sid in train_source_ids:
        r = target_ratios[sid]
        n = max(n_per_source[sid], 1)
        weights.append(r / n)
    weights = torch.tensor(weights, dtype=torch.double)
    return torch.utils.data.WeightedRandomSampler(
        weights, num_samples=num_samples, replacement=replacement
    )


def get_uie_train_val_datasets_and_sampler(args, train_ratio=0.9):
    """One-shot helper for Task 7: mixed UIE train/val datasets and weighted train sampler.
    Returns:
        train_dataset: UnderwaterDataset for training
        val_dataset: UnderwaterDataset for validation (can be empty)
        train_sampler: WeightedRandomSampler for train (use with DataLoader(..., sampler=train_sampler, shuffle=False)
    """
    train_samples, val_samples, train_source_ids = build_uie_mixed_pool_and_split(
        args, train_ratio=train_ratio, seed=42
    )
    train_dataset = UnderwaterDataset(train_samples, args.patch_size)
    val_dataset = UnderwaterDataset(val_samples, args.patch_size)
    train_sampler = build_uie_weighted_sampler(
        train_source_ids, num_samples=len(train_dataset), replacement=True
    )
    return train_dataset, val_dataset, train_sampler

    
class AdaIRTrainDataset(Dataset):
    def __init__(self, args):
        super(AdaIRTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5, 'enhance' : 6}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'deblur' in self.de_type:
            self._init_deblur_ids()
        if 'enhance' in self.de_type:
            self._init_enhance_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        ref_file = self.args.data_file_dir + "noisy/denoise.txt"
        temp_ids = []
        temp_ids+= [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.data_file_dir + "hazy/hazy_outside.txt"
        temp_ids+= [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]

        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_deblur_ids(self):
        temp_ids = []

        image_list = os.listdir(os.path.join(self.args.gopro_dir, 'blur/'))
        temp_ids = image_list
        self.deblur_ids = [{"clean_id" : x,"de_type":5} for x in temp_ids]
        self.deblur_ids = self.deblur_ids * 5
        self.deblur_counter = 0
        self.num_deblur = len(self.deblur_ids)
        print('Total Blur Ids : {}'.format(self.num_deblur))

    def _init_enhance_ids(self):
        temp_ids = []
        image_list = os.listdir(os.path.join(self.args.enhance_dir, 'low/'))
        temp_ids = image_list
        self.enhance_ids= [{"clean_id" : x,"de_type":6} for x in temp_ids]
        self.enhance_ids = self.enhance_ids * 20
        self.num_enhance = len(self.enhance_ids)
        print('Total enhance Ids : {}'.format(self.num_enhance))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"
        temp_ids+= [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 120

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name


    def _get_deblur_name(self, deblur_name):
        gt_name = deblur_name.replace("blur", "sharp")
        return gt_name
    

    def _get_enhance_name(self, enhance_name):
        gt_name = enhance_name.replace("low", "gt")
        return gt_name


    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids
        if "deblur" in self.de_type:
            self.sample_ids += self.deblur_ids
        if "enhance" in self.de_type:
            self.sample_ids += self.enhance_ids

        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 5:
                # Deblur with Gopro set
                degrad_img = crop_img(np.array(Image.open(os.path.join(self.args.gopro_dir, 'blur/', sample["clean_id"])).convert('RGB')), base=16)
                clean_img = crop_img(np.array(Image.open(os.path.join(self.args.gopro_dir, 'sharp/', sample["clean_id"])).convert('RGB')), base=16)
                clean_name = self._get_deblur_name(sample["clean_id"])
            elif de_id == 6:
                # Enhancement with LOL training set
                degrad_img = crop_img(np.array(Image.open(os.path.join(self.args.enhance_dir, 'low/', sample["clean_id"])).convert('RGB')), base=16)
                clean_img = crop_img(np.array(Image.open(os.path.join(self.args.enhance_dir, 'gt/', sample["clean_id"])).convert('RGB')), base=16)
                clean_name = self._get_enhance_name(sample["clean_id"])

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)


        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img
    def tile_degrad(input_,tile=128,tile_overlap =0):
        sigma_dict = {0:0,1:15,2:25,3:50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = in_patch
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)

        restored = torch.clamp(restored, 0, 1)
        return restored
    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1, 'deblur': 2, 'enhance': 3}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 2:
            self.ids = []
            name_list = os.listdir(self.args.gopro_path +'input/')
            self.ids += [self.args.gopro_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 3:
            self.ids = []
            name_list = os.listdir(self.args.enhance_path + 'input/')
            self.ids += [self.args.enhance_path + 'input/' + id_ for id_ in name_list]


        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        elif self.task_idx == 2:
            gt_name = degraded_name.replace("input", "target")

        elif self.task_idx == 3:
            gt_name = degraded_name.replace("input", "target")

        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
    
