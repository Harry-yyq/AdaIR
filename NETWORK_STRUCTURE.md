# AdaIR 网络结构（当前代码）

默认参数：`dim=48`, `num_blocks=[4,6,6,8]`, `num_refinement_blocks=4`, `heads=[1,2,4,8]`, `decoder=True`。

---

## 一、当前数据流（与 forward 一致）

```
inp_img (B,3,H,W)   ← 唯一输入，无外部深度
    │
    ▼
depth_map = _estimate_depth(inp_img)   ← 内部：灰度化 → DepthAnythingV2(no_grad) → min-max 归一化 [0,1]，(B,1,H,W)；若无 DAV2 则全 0.5
    │
    ▼
patch_embed(inp_img)  →  feat (B,48,H,W)
    │
    ▼
encoder_level1(feat)   →  out_enc_level1 (B,48,H,W)
    │
    ▼
down1_2  →  inp_enc_level2 (B,96,H/2,W/2)
    │
    ▼
encoder_level2  →  out_enc_level2 (B,96,H/2,W/2)
    │
    ▼
down2_3  →  inp_enc_level3 (B,192,H/4,W/4)
    │
    ▼
encoder_level3  →  out_enc_level3 (B,192,H/4,W/4)
    │
    ▼
down3_4  →  inp_enc_level4 (B,384,H/8,W/8)
    │
    ▼
latent(inp_enc_level4)  →  feat (B,384,H/8,W/8)
    │
    ▼
eccm(feat, depth_map)  →  feat (B,384,H/8,W/8)   ← 深层色彩校正，depth 内部 interpolate 到 H/8×W/8
    │
    ▼
若 decoder:  fre1(inp_img, feat, depth_map)  →  feat (B,384,H/8,W/8)
    │
    ▼
up4_3(feat)  →  inp_dec_level3 (B,192,H/4,W/4)
concat(inp_dec_level3, out_enc_level3)  →  reduce_chan_level3  →  (B,192,H/4,W/4)
    │
    ▼
decoder_level3  →  out_dec_level3
若 decoder:  fre2(inp_img, out_dec_level3, depth_map)
    │
    ▼
up3_2  →  concat  →  reduce_chan_level2  →  decoder_level2  →  out_dec_level2
若 decoder:  fre3(inp_img, out_dec_level2, depth_map)
    │
    ▼
up2_1  →  concat  →  decoder_level1  →  out_dec_level1
    │
    ▼
refinement(out_dec_level1)  →  output(·) + inp_img  →  out (B,3,H,W)
```

---

## 二、模块层级（当前实现）

| 阶段 | 子模块 | 说明 | 输入 → 输出 (B,C,H,W) |
|------|--------|------|------------------------|
| 深度 | **depth_estimator** | DepthAnythingV2 ViT-Small（冻结），灰度图→深度→[0,1] | 内部 _estimate_depth(inp_img) → (B,1,H,W) |
| 入口 | **patch_embed** | OverlapPatchEmbed(3→48), 3×3 conv | (B,3,H,W) → (B,48,H,W) |
| Enc L1 | **encoder_level1** | 4× TransformerBlock(dim=48, heads=1) | (B,48,H,W) |
| ↓ | **down1_2** | Downsample: Conv3×3 + PixelUnshuffle(2) | (B,48,H,W) → (B,96,H/2,W/2) |
| Enc L2 | **encoder_level2** | 6× TransformerBlock(dim=96, heads=2) | (B,96,H/2,W/2) |
| ↓ | **down2_3** | Downsample | (B,96) → (B,192,H/4,W/4) |
| Enc L3 | **encoder_level3** | 6× TransformerBlock(dim=192, heads=4) | (B,192,H/4,W/4) |
| ↓ | **down3_4** | Downsample | (B,192) → (B,384,H/8,W/8) |
| 瓶颈 | **latent** | 8× TransformerBlock(dim=384, heads=8) | (B,384,H/8,W/8) |
| 瓶颈 | **eccm** | EarlyColorCorrectionModule(384)，始终用内部 depth_map | (B,384,H/8,W/8) → (B,384,H/8,W/8)，depth 内部 interpolate 到 H/8×W/8 |
| 瓶颈 | **fre1** | FreModule(384)，使用内部 depth_map 做 sft_low/sft_high | (B,384,H/8,W/8) |
| 上采样 | **up4_3** | Upsample | (B,384) → (B,192,H/4,W/4) |
| | **reduce_chan_level3** | Conv2d(384→192, 1×1) | concat 后 384→192 |
| Dec L3 | **decoder_level3** | 6× TransformerBlock(192) | (B,192,H/4,W/4) |
| | **fre2** | FreModule(192), 可选 depth_map | (B,192,H/4,W/4) |
| 上采样 | **up3_2** | Upsample | (B,192) → (B,96,H/2,W/2) |
| | **reduce_chan_level2** | Conv2d(192→96, 1×1) | |
| Dec L2 | **decoder_level2** | 6× TransformerBlock(96) | (B,96,H/2,W/2) |
| | **fre3** | FreModule(96), 可选 depth_map | (B,96,H/2,W/2) |
| 上采样 | **up2_1** | Upsample | (B,96) → (B,48,H,W) |
| Dec L1 | **decoder_level1** | 4× TransformerBlock(48) | (B,48,H,W) |
| 精修 | **refinement** | 4× TransformerBlock(48) | (B,48,H,W) |
| 输出 | **output** | Conv2d(48→3, 3×3) + 残差加 inp_img | (B,3,H,W) |

---

## 三、模块版结构（按类/组件分层）

按 **类（Module）** 组织，便于查阅每个组件的输入输出与内部子模块。

### 3.1 主网络：AdaIR

```
AdaIR
├── depth_estimator      DepthAnythingV2(encoder='vits', features=64, out_channels=[48,96,192,384])，冻结，可选
├── patch_embed          OverlapPatchEmbed(3, 48)
├── encoder_level1      Sequential[ 4 × TransformerBlock(48, heads=1) ]
├── down1_2              Downsample(48)
├── encoder_level2      Sequential[ 6 × TransformerBlock(96, heads=2) ]
├── down2_3              Downsample(96)
├── encoder_level3      Sequential[ 6 × TransformerBlock(192, heads=4) ]
├── down3_4              Downsample(192)
├── latent              Sequential[ 8 × TransformerBlock(384, heads=8) ]
├── eccm                 EarlyColorCorrectionModule(384)
├── fre1                 FreModule(384)   [decoder=True 时存在]
├── up4_3                Upsample(384)
├── reduce_chan_level3   Conv2d(384, 192, 1)
├── decoder_level3      Sequential[ 6 × TransformerBlock(192) ]
├── fre2                 FreModule(192)
├── up3_2                Upsample(192)
├── reduce_chan_level2   Conv2d(192, 96, 1)
├── decoder_level2      Sequential[ 6 × TransformerBlock(96) ]
├── fre3                 FreModule(96)
├── up2_1                Upsample(96)
├── decoder_level1      Sequential[ 4 × TransformerBlock(48) ]
├── refinement          Sequential[ 4 × TransformerBlock(48) ]
└── output              Conv2d(48, 3, 3)
```

### 3.2 EarlyColorCorrectionModule (ECCM)

- **输入**：`x` (B, C, H, W)，`depth_map` (B, 1, H_in, W_in)；内部将 depth 插值到 (H,W)。
- **输出**：`x * (1 + tanh(depth_projector(depth_map)))`，与 `x` 同形状；零初始化保证初始时 ≈ x。

```
EarlyColorCorrectionModule(in_channels=384)
└── depth_projector   Sequential
    ├── Conv2d(1, 192, 3, padding=1)
    ├── GELU
    └── Conv2d(192, 384, 3, padding=1)   # 零初始化
```

### 3.3 TransformerBlock（Encoder/Decoder 共用）

```
TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
├── norm1    LayerNorm(dim)
├── attn     Attention(dim, num_heads)     # MDTA
├── norm2    LayerNorm(dim)
└── ffn      FeedForward(dim, ffn_expansion_factor)
```

- **Attention (MDTA)**：Conv 1×1 → qkv，dwconv 3×3，chunk → q,k,v，多头 transposed attention → Conv 1×1。
- **FeedForward (GDFN)**：Conv 1×1 扩展 → dwconv 3×3 → GELU*gate → Conv 1×1。

### 3.4 FreModule（频率调制 + 深度引导）

- **输入**：`x` 原图 (B,3,·,·)，`y` 当前层特征 (B,C,H,W)，`depth` (B,1,·,·) 可选。
- **流程**：原图 FFT 高/低频 → 可选 SFT 深度调制 → channel_cross_l/h → frequency_refine → channel_cross_agg → 加权与 y 融合。

```
FreModule(dim, num_heads, in_dim=3)
├── conv, conv1           Conv2d(3, dim, 3)
├── score_gen             Conv2d(2, 2, 7)
├── rate_conv             Sequential  # 自适应阈值
├── para1, para2          Parameter   # 输出融合权重
├── channel_cross_l       Chanel_Cross_Attention(dim)
├── channel_cross_h       Chanel_Cross_Attention(dim)
├── channel_cross_agg     Chanel_Cross_Attention(dim)
├── frequency_refine      FreRefine(dim)
├── sft_low               SpatialFeatureTransform(dim)
└── sft_high              SpatialFeatureTransform(dim)
```

### 3.5 SpatialFeatureTransform (SFT)

- **输入**：depth (B,1,H,W)，feat (B,C,H,W)。
- **输出**：`feat * (1 + scale) + shift`；最后一层零初始化，初始时 out=feat。

```
SpatialFeatureTransform(channels)
├── scale_conv1, scale_act, scale_conv2   # 零初始化 scale_conv2
└── shift_conv1, shift_act, shift_conv2   # 零初始化 shift_conv2
```

### 3.6 其他基础模块

| 类名 | 作用 |
|------|------|
| **OverlapPatchEmbed** | Conv2d(in_c, embed_dim, 3×3, pad=1) |
| **Downsample** | Conv2d(n_feat, n_feat//2, 3) + PixelUnshuffle(2) |
| **Upsample** | Conv2d(n_feat, n_feat*2, 3) + PixelShuffle(2) |
| **Chanel_Cross_Attention** | 通道维 Q from x, KV from y，cross attention |
| **FreRefine** | SpatialGate + ChannelGate + proj 融合 high/low |
| **LayerNorm** | BiasFree / WithBias LayerNorm |
| **Attention** | MDTA |
| **FeedForward** | GDFN |

---

## 四、分辨率与通道（示例 H=W=256）

| 阶段 | 分辨率 | 通道 |
|------|--------|------|
| 输入 | 256×256 | 3 |
| patch_embed | 256×256 | 48 |
| L1 后 | 256×256 | 48 |
| L2 后 | 128×128 | 96 |
| L3 后 | 64×64 | 192 |
| latent 后 | 32×32 | 384 |
| eccm（内部 depth） | 32×32 | 384 |
| fre1 后 | 32×32 | 384 |
| 上采样 + L3 解码 | 64×64 | 192 |
| fre2 后 | 64×64 | 192 |
| 上采样 + L2 解码 | 128×128 | 96 |
| fre3 后 | 128×128 | 96 |
| 上采样 + L1 解码 + refinement | 256×256 | 48 → 3 |

---

## 五、forward 接口

```python
# 仅需输入图像；深度在内部由 DepthAnythingV2 估计，供 ECCM 与 FreModule 使用
out = net(inp_img)   # inp_img: (B,3,H,W)；可选 noise_emb 未使用
```
