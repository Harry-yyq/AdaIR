# UG-AdaIR 网络结构（当前代码）

默认参数：`dim=48`, `num_blocks=[4,6,6,8]`, `num_refinement_blocks=4`, `heads=[1,2,4,8]`, `decoder=True`。

---

## 一、整体数据流

```
inp_img (B,3,H,W) [+ depth_map (B,1,H,W) 可选]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Encoder (下采样 3 次：H/2, H/4, H/8)                             │
│  patch_embed → L1 → down → L2 → down → L3 → down → latent        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Bottleneck + 频率调制 (FMoM)                                    │
│  latent → fre1(inp_img, latent, depth_map) → 上采样              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Decoder (上采样 + skip + FMoM ×2)                               │
│  up→concat→L3→fre2(...) → up→concat→L2 → up→concat→L1           │
│  → refinement → output + inp_img                                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
out (B,3,H,W)
```

---

## 二、模块层级

### 1. AdaIR (主网络)

| 子模块 | 说明 | 通道/分辨率 |
|--------|------|-------------|
| **patch_embed** | OverlapPatchEmbed(3→48), 3×3 conv | 输入 (B,3,H,W) → (B,48,H,W) |
| **encoder_level1** | 4× TransformerBlock(dim=48, heads=1) | (B,48,H,W) |
| **down1_2** | Downsample: Conv3×3 + PixelUnshuffle(2) | (B,48,H,W) → (B,96,H/2,W/2) |
| **encoder_level2** | 6× TransformerBlock(dim=96, heads=2) | (B,96,H/2,W/2) |
| **down2_3** | Downsample | (B,96) → (B,192,H/4,W/4) |
| **encoder_level3** | 6× TransformerBlock(dim=192, heads=4) | (B,192,H/4,W/4) |
| **down3_4** | Downsample | (B,192) → (B,384,H/8,W/8) |
| **latent** | 8× TransformerBlock(dim=384, heads=8) | (B,384,H/8,W/8) |
| **fre1** | FreModule(384), 可选 depth_map | (B,384,H/8,W/8) |
| **up4_3** | Upsample | (B,384) → (B,192,H/4,W/4) |
| **reduce_chan_level3** | Conv2d(384→192, 1×1) | 与 skip 拼接后 384→192 |
| **decoder_level3** | 6× TransformerBlock(192) | (B,192,H/4,W/4) |
| **fre2** | FreModule(192), 可选 depth_map | (B,192,H/4,W/4) |
| **up3_2** | Upsample | (B,192) → (B,96,H/2,W/2) |
| **reduce_chan_level2** | Conv2d(192→96, 1×1) |  |
| **decoder_level2** | 6× TransformerBlock(96) | (B,96,H/2,W/2) |
| **fre3** | FreModule(96), 可选 depth_map | (B,96,H/2,W/2) |
| **up2_1** | Upsample | (B,96) → (B,48,H,W) |
| **decoder_level1** | 4× TransformerBlock(48) | (B,48,H,W) |
| **refinement** | 4× TransformerBlock(48) | (B,48,H,W) |
| **output** | Conv2d(48→3, 3×3) + 残差加 inp_img | (B,3,H,W) |

---

### 2. TransformerBlock（Encoder/Decoder 共用）

```
x → LayerNorm → Attention(MDTA) → (+) → LayerNorm → FeedForward(GDFN) → (+) → out
```

- **Attention (MDTA)**：Conv 1×1 出 qkv，dwconv 3×3，chunk 成 q,k,v → 多头 transposed attention → Conv 1×1。
- **FeedForward (GDFN)**：Conv 1×1 扩展 → dwconv 3×3 → GELU*gate → Conv 1×1。

---

### 3. FreModule（频率调制 + 深度引导，UG-AdaIR 扩展）

- **输入**：`x` 原图 (B,3,·,·)，`y` 当前层特征 (B,C,H,W)，`depth` (B,1,·,·) 可选。
- **流程**：
  1. 将 `x` 插值到 (H,W)，经 **fft()** 拆成 high_feature、low_feature。
  2. 若 `depth is not None`：
     - 将 depth 插值到 (H,W)，得到 depth_deep、depth_shallow=1-depth。
     - **sft_low**(depth_deep, low_feature) → low_mod，再乘 depth_deep。
     - **sft_high**(depth_shallow, high_feature) → high_mod，再乘 depth_shallow。
  3. **channel_cross_l**(high_feature, y)、**channel_cross_h**(low_feature, y)。
  4. **frequency_refine**(low, high) → agg。
  5. **channel_cross_agg**(y, agg) → out，返回 `out*para1 + y*para2`。

子模块：

| 子模块 | 说明 |
|--------|------|
| conv1 | Conv2d(3, C, 3×3)，用于 fft 分支 |
| rate_conv | 自适应阈值生成 mask（C→C/8→2, Sigmoid） |
| fft | FFT2 → 高/低频 mask 分离 → IFFT2 → abs → high, low |
| **sft_low** | SpatialFeatureTransform(C)，深度调制低频 |
| **sft_high** | SpatialFeatureTransform(C)，(1-depth) 调制高频 |
| channel_cross_l / channel_cross_h / channel_cross_agg | Chanel_Cross_Attention |
| frequency_refine | FreRefine：SpatialGate + ChannelGate + 融合 |

---

### 4. SpatialFeatureTransform (SFT，UG-AdaIR 新增)

- **输入**：depth (B,1,H,W)，feat (B,C,H,W)。
- **输出**：`feat * (1 + scale) + shift`，其中 scale、shift 由两条分支从 depth 生成，**最后一层 Conv 零初始化**，初始时 out=feat。
- **结构**：
  - scale 分支：Conv2d(1→C, 3×3) → LeakyReLU → Conv2d(C→C, 3×3)，零初始化。
  - shift 分支：Conv2d(1→C, 3×3) → LeakyReLU → Conv2d(C→C, 3×3)，零初始化。

---

### 5. 其他基础块

| 模块 | 作用 |
|------|------|
| OverlapPatchEmbed | Conv2d(in_c, embed_dim, 3×3, pad=1) |
| Downsample | Conv2d(n_feat, n_feat//2, 3×3) + PixelUnshuffle(2)，分辨率/2，通道/2 |
| Upsample | Conv2d(n_feat, n_feat*2, 3×3) + PixelShuffle(2)，分辨率×2，通道/2 |
| Chanel_Cross_Attention | q 来自 x，kv 来自 y，通道维 cross attention |
| FreRefine | SpatialGate(high) 加权 high，ChannelGate(low) 加权 low，相加后 1×1 conv |

---

## 三、参数量与分辨率对应（示例 H=W=256）

| 阶段 | 分辨率 | 通道 dim |
|------|--------|----------|
| 输入 | 256×256 | 3 |
| L1 后 | 256×256 | 48 |
| L2 后 | 128×128 | 96 |
| L3 后 | 64×64 | 192 |
| latent 后 | 32×32 | 384 |
| fre1 后 | 32×32 | 384 |
| 上采样 + L3 解码 | 64×64 | 192 |
| fre2 后 | 64×64 | 192 |
| 上采样 + L2 解码 | 128×128 | 96 |
| fre3 后 | 128×128 | 96 |
| 上采样 + L1 解码 + refinement | 256×256 | 48 → 3 |

---

## 四、forward 接口

```python
# 训练 UIE（带深度）
out = net(inp_img, depth_map)   # depth_map: (B,1,H,W), 0~1

# 多任务 / 测试（无深度）
out = net(inp_img)              # depth_map=None，FMoM 中不执行 SFT 与深度门控
```
