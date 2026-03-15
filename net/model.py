## AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation
## Yuning Cui, Syed Waqas Zamir, Salman Khan, Alois Knoll, Mubarak Shah, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2403.14614


import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange

# DepthAnythingV2: 若项目下存在 Depth-Anything-V2 仓库，先加入 path 再导入
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dav2_dir = os.path.join(_project_root, "Depth-Anything-V2")
if os.path.isdir(_dav2_dir) and _dav2_dir not in sys.path:
    sys.path.insert(0, _dav2_dir)
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    _HAS_DAV2 = True
except ImportError as _e_dav2:
    DepthAnythingV2 = None
    _HAS_DAV2 = False
    _DAV2_IMPORT_ERROR = _e_dav2
_DAV2_WARNED = False


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Channel-Wise Cross Attention (CA)
class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)


        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
    

##########################################################################
## H-L Unit
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max = torch.max(x,1,keepdim=True)[0]
        mean = torch.mean(x,1,keepdim=True)
        scale = torch.cat((max, mean), dim=1)
        scale =self.spatial(scale)
        scale = F.sigmoid(scale)
        return scale

##########################################################################
## L-H Unit
class ChannelGate(nn.Module):
    def __init__(self, dim):
        super(ChannelGate, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.max = nn.AdaptiveMaxPool2d((1,1))

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim//16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim//16, dim, 1, bias=False)
        )

    def forward(self, x):
        avg = self.mlp(self.avg(x))
        max = self.mlp(self.max(x))

        scale = avg + max
        scale = F.sigmoid(scale)
        return scale


##########################################################################
## UG-AdaIR: 早期色彩校正模块 (ECCM)，参考 DarkIR，零初始化保证初始恒等
ECCM_STATS_LOG_DIR = "output"
ECCM_STATS_LOG_FILE = "output/eccm_scale_stats.log"
ECCM_STATS_PRINT_EVERY = 100


class EarlyColorCorrectionModule(nn.Module):
    def __init__(self, in_channels=48):
        super().__init__()
        # 利用深度图生成空间-通道不对称的调制权重
        self.depth_projector = nn.Sequential(
            nn.Conv2d(1, in_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1)
        )
        # 【强制零初始化】：保证初始状态下输出为0，网络等价于原版，防止Loss爆炸
        nn.init.constant_(self.depth_projector[-1].weight, 0)
        nn.init.constant_(self.depth_projector[-1].bias, 0)
        # ECCM 权值统计探针：前向计数，每 N 次打印并写入日志
        self._eccm_forward_count = 0

    def forward(self, x, depth_map):
        # 尺寸对齐
        if depth_map.shape[-2:] != x.shape[-2:]:
            depth_map = F.interpolate(depth_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # 限制放大倍数，Tanh 的输出在 -1 到 1 之间
        scale = torch.tanh(self.depth_projector(depth_map))
        # 乘法门控残差：允许最大放大 2 倍，最小变为 0
        corrected_x = x * (1.0 + scale)

        # ---------- ECCM 权值统计探针：scale 的 Mean/Max 绝对值，每 N 次打印并落盘 ----------
        self._eccm_forward_count += 1
        if self._eccm_forward_count % ECCM_STATS_PRINT_EVERY == 0:
            with torch.no_grad():
                scale_abs = scale.detach().abs()
                mean_abs = scale_abs.mean().item()
                max_abs = scale_abs.max().item()
            msg = "[ECCM scale] step_approx={} | mean_abs={:.6f} | max_abs={:.6f}".format(
                self._eccm_forward_count, mean_abs, max_abs
            )
            try:
                import os
                os.makedirs(ECCM_STATS_LOG_DIR, exist_ok=True)
                with open(ECCM_STATS_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

        return corrected_x, scale


##########################################################################
## UG-AdaIR: 深度引导空间特征变换 (SFT)，Zero-Conv 初始化保证初始恒等
DEBUG_MODE = True


class SpatialFeatureTransform(nn.Module):
    """
    输入: depth (B,1,H,W), feat (B,C,H,W)
    输出: out = feat * (1 + scale) + shift
    生成 scale/shift 的最后一层 Conv 零初始化，初始时 out = feat。
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.is_sft = True

        self.scale_conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.scale_act = nn.LeakyReLU(0.1, inplace=True)
        self.scale_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.shift_conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.shift_act = nn.LeakyReLU(0.1, inplace=True)
        self.shift_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self._init_zero()

    def _init_zero(self):
        nn.init.zeros_(self.scale_conv2.weight)
        nn.init.zeros_(self.scale_conv2.bias)
        nn.init.zeros_(self.shift_conv2.weight)
        nn.init.zeros_(self.shift_conv2.bias)

    def forward(self, depth, feat):
        B, C, H, W = feat.shape
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)
        depth = depth.clamp(0.0, 1.0)

        scale = self.scale_conv2(self.scale_act(self.scale_conv1(depth)))
        shift = self.shift_conv2(self.shift_act(self.shift_conv1(depth)))

        if DEBUG_MODE:
            assert feat.shape[-2:] == scale.shape[-2:], "SFT feat/scale H,W mismatch"
            assert feat.shape == shift.shape, "SFT feat/shift shape mismatch"

        return feat * (1.0 + scale) + shift


##########################################################################
## Frequency Modulation Module (FMoM)
class FreRefine(nn.Module):
    def __init__(self, dim):
        super(FreRefine, self).__init__()

        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, low, high):
        spatial_weight = self.SpatialGate(high)
        channel_weight = self.ChannelGate(low)
        high = high * channel_weight
        low = low * spatial_weight

        out = low + high
        out = self.proj(out)
        return out
    
##########################################################################
## Adaptive Frequency Learning Block (AFLB)
class FreModule(nn.Module):
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(FreModule, self).__init__()

        self.conv = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.score_gen = nn.Conv2d(2, 2, 7, padding=3)

        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.channel_cross_l = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_h = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_agg = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)

        self.frequency_refine = FreRefine(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim//8, 2, 1, bias=False),
        )
        self.sft_low = SpatialFeatureTransform(dim)
        self.sft_high = SpatialFeatureTransform(dim)

    def forward(self, x, y, depth=None):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H, W), mode='bilinear')

        high_feature, low_feature = self.fft(x)
        high_mod = high_feature
        low_mod = low_feature

        if depth is not None:
            depth_deep = F.interpolate(depth, size=low_feature.shape[-2:], mode='bilinear', align_corners=False).clamp(0.0, 1.0)
            depth_shallow = 1.0 - depth_deep
            low_mod = self.sft_low(depth_deep, low_feature) * depth_deep
            high_mod = self.sft_high(depth_shallow, high_feature) * depth_shallow

        high_mod = self.channel_cross_l(high_mod, y)
        low_mod = self.channel_cross_h(low_mod, y)
        agg = self.frequency_refine(low_mod, high_mod)
        out = self.channel_cross_agg(y, agg)

        return out * self.para1 + y * self.para2

    def shift(self, x):
        '''shift FFT feature map to center'''
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2,3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h ,w = x.shape
        return torch.roll(x, shifts=(-int(h/2), -int(w/2)), dims=(2,3))

    def fft(self, x, n=128):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = self.rate_conv(threshold).sigmoid()

        for i in range(mask.shape[0]):
            h_ = (h//n * threshold[i,0,:,:]).int()
            w_ = (w//n * threshold[i,1,:,:]).int()

            mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1

        fft = torch.fft.fft2(x, norm='forward', dim=(-2,-1))
        fft = self.shift(fft)
        
        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2,-1))
        high = torch.abs(high)

        fft_low = fft * mask

        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2,-1))
        low = torch.abs(low)

        return high, low


##########################################################################
##---------- AdaIR -----------------------

class AdaIR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias', 
        decoder = True,
    ):

        super(AdaIR, self).__init__()

        # ---------------- 深度估计子网（DepthAnythingV2 ViT-Small，冻结） ----------------
        self.depth_estimator = None
        if _HAS_DAV2:
            try:
                self.depth_estimator = DepthAnythingV2(
                    encoder='vits',
                    features=64,
                    out_channels=[48, 96, 192, 384],
                )
                # 优先使用相对路径：项目根/ckpt/DepthAnythingv2/xxx.pth（与 model.py 所在 net/ 同级）
                _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ckpt_path = os.path.join(_root, "ckpt", "DepthAnythingv2", "depth_anything_v2_vits.pth")
                ckpt_path = os.path.normpath(ckpt_path)
                if not os.path.isfile(ckpt_path):
                    raise FileNotFoundError("DepthAnythingV2 权重文件不存在: {}".format(ckpt_path))
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                if not state or len(state) < 10:
                    raise RuntimeError("加载的 state 为空或 key 过少(len={})，请检查 ckpt 是否完整".format(len(state) if state else 0))
                self.depth_estimator.load_state_dict(state, strict=True)
                # 加载权重后打印一小部分权重值：Norm1.weight 均值应在 0.8~1.0 左右，若接近 0 说明加载失败
                try:
                    m = self.depth_estimator.pretrained
                    if hasattr(m, "blocks"):
                        first_weight = m.blocks[0].norm1.weight
                    elif hasattr(m, "model") and hasattr(m.model, "blocks"):
                        first_weight = m.model.blocks[0].norm1.weight
                    else:
                        first_weight = next(self.depth_estimator.parameters())
                    w_mean = first_weight.abs().mean().item()
                    print(f"[DepthAnythingV2] 权重加载后 Norm1.weight 均值: {w_mean:.6f} (正常约 0.8~1.0，若≈0 则加载失败)")
                except Exception as e:
                    p = next(self.depth_estimator.parameters())
                    print(f"[DepthAnythingV2] 权重加载后 首参数 abs mean: {p.abs().mean().item():.6f} (fallback)", e)
                for p in self.depth_estimator.parameters():
                    p.requires_grad = False
                self.depth_estimator.eval()
                print("[DepthAnythingV2] ViT-Small loaded and frozen from:", ckpt_path)
            except Exception as e:
                print("[DepthAnythingV2] 初始化失败，将禁用深度估计子网:", e)
                self.depth_estimator = None
        else:
            global _DAV2_WARNED
            if not _DAV2_WARNED:
                print("[DepthAnythingV2] 未启用：depth_anything_v2 导入失败，深度图将使用常数 0.5。")
                print("  导入错误:", _DAV2_IMPORT_ERROR)
                print("  解决：将 Depth-Anything-V2 仓库 clone 到项目下并安装依赖，或 pip 安装后确保可 import depth_anything_v2.dpt")
                _DAV2_WARNED = True

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder
        
        if self.decoder:
            self.fre1 = FreModule(dim*2**3, num_heads=heads[2], bias=bias)
            self.fre2 = FreModule(dim*2**2, num_heads=heads[2], bias=bias)
            self.fre3 = FreModule(dim*2**1, num_heads=heads[2], bias=bias)            

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.eccm = EarlyColorCorrectionModule(in_channels=384)

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) 
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1)) 

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def train(self, mode=True):
        """训练时保持 depth_estimator 始终为 eval，避免 Dropout 等影响深度估计稳定性。"""
        super(AdaIR, self).train(mode)
        if self.depth_estimator is not None:
            self.depth_estimator.eval()
        return self

    # DepthAnythingV2 ViT 的 patch 尺寸，输入高宽需为其倍数
    DEPTH_PATCH = 14

    def _estimate_depth(self, inp_img):
        """
        使用 DepthAnythingV2 估计单目深度，并归一化到 [0,1]。
        若 depth_estimator 不可用，则返回全 0.5 的“中性深度”图。
        会对输入做 pad 使 H/W 为 patch 倍数，推理后再裁剪回原始尺寸。
        """
        b, c, h, w = inp_img.shape
        device = inp_img.device
        if self.depth_estimator is None:
            return torch.full((b, 1, h, w), 0.5, dtype=inp_img.dtype, device=device)

        # 灰度化后复制为 3 通道；DepthAnythingV2 官方使用 ImageNet 标准化，输入为 [0,1] RGB 再 (x-mean)/std
        r, g, bch = inp_img[:, 0:1], inp_img[:, 1:2], inp_img[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * bch            # (B,1,H,W)
        rgb_gray = gray.repeat(1, 3, 1, 1)                    # (B,3,H,W)，[0,1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_gray.device, dtype=rgb_gray.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb_gray.device, dtype=rgb_gray.dtype).view(1, 3, 1, 1)
        rgb_norm = (rgb_gray - mean) / std

        # 使 H、W 为 DEPTH_PATCH 的倍数，避免 DAV2 patch_embed 断言失败
        p = self.DEPTH_PATCH
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h > 0 or pad_w > 0:
            rgb_norm = torch.nn.functional.pad(rgb_norm, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, h_pad, w_pad = rgb_norm.shape

        with torch.no_grad():
            depth_raw = self.depth_estimator(rgb_norm)        # (B, H_pad, W_pad)
        depth_raw = depth_raw.unsqueeze(1)                    # (B,1,H_pad,W_pad)
        if pad_h > 0 or pad_w > 0:
            depth_raw = depth_raw[:, :, :h, :w]

        # 归一化到 [0,1]，在 batch 维度内做 min-max
        d_min = depth_raw.amin(dim=[2, 3], keepdim=True)
        d_max = depth_raw.amax(dim=[2, 3], keepdim=True)
        denom = (d_max - d_min).clamp(min=1e-6)
        depth_norm = (depth_raw - d_min) / denom
        return depth_norm.clamp(0.0, 1.0)

    def forward(self, inp_img, noise_emb=None, return_aux=False):
        # 内部估计深度图，供 ECCM 和 FreModule 使用
        depth_map = self._estimate_depth(inp_img)

        # 深度图强制审计哨兵：仅第 1 次前向时打印，避免刷屏
        if not hasattr(self, '_checked_depth'):
            with torch.no_grad():
                d_min = depth_map.min().item()
                d_max = depth_map.max().item()
                d_mean = depth_map.mean().item()
                d_std = depth_map.std().item()
            print(f"\n[Depth Debug] depth_map min: {d_min:.4f}, max: {d_max:.4f}, mean: {d_mean:.4f}, std: {d_std:.6f}")
            if d_std < 1e-5:
                print("🚨 警告：depth_map 方差极小，可能是全黑/常数图！请检查 DepthAnything 权重加载！")
                if abs(d_mean - 0.5) < 0.01:
                    print("  → 当前为常数 0.5，说明未使用 DepthAnythingV2，走的是「无深度估计」兜底逻辑（self.depth_estimator 为 None）。")
                    print("  → 请检查：1) 是否安装/可导入 depth_anything_v2；2) ckpt/DepthAnythingv2/depth_anything_v2_vits.pth 是否存在；3) 模型创建时是否有 [DepthAnythingV2] 初始化失败 的报错。")
            else:
                print("  (若 mean≈0.5 且 std 不为 0，则数值正常；若可视化仍像噪声，请检查可视化是否按 [0,1] 映射、未过度拉伸)")
            self._checked_depth = True

        aux = {} if return_aux else None

        feat = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(feat)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)
        feat = self.latent(inp_enc_level4)
        if depth_map is not None:
            feat, eccm_scale = self.eccm(feat, depth_map)
            if return_aux:
                aux["depth_map"] = depth_map
                aux["eccm_scale"] = eccm_scale
        else:
            if return_aux:
                aux["depth_map"] = None
                aux["eccm_scale"] = None

        if self.decoder:
            feat = self.fre1(inp_img, feat, depth_map)
      
        inp_dec_level3 = self.up4_3(feat)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        if self.decoder:
            out_dec_level3 = self.fre2(inp_img, out_dec_level3, depth_map)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            out_dec_level2 = self.fre3(inp_img, out_dec_level2, depth_map)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        if return_aux:
            return out_dec_level1, aux
        return out_dec_level1
    
