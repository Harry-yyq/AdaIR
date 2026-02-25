# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

AdaIR (Adaptive All-in-One Image Restoration) is a pure Python/PyTorch ML research codebase (ICLR 2025). There is **no web UI, no database, no Docker, no test framework, and no linter** configured. The two entry points are `train.py` (multi-GPU DDP training via PyTorch Lightning) and `test.py` (evaluation/inference).

### Environment notes

- The original `env.yaml` specifies Python 3.8.11 + PyTorch 1.13.1 + CUDA 11.6 via **conda**. Since the Cloud VM has Python 3.12 and no conda, we install dependencies via pip with recent compatible versions (CPU-only PyTorch).
- **No GPU is available** in the Cloud VM. The code's `train.py` and `test.py` call `.cuda()` and `accelerator="gpu"`, so full training/testing cannot run without a GPU. However, the model can be instantiated and run forward passes on CPU for verification.
- All Python imports and module-level code work correctly on CPU with the pip-installed dependencies.

### How to verify the environment

```bash
cd /workspace
python3 -c "from net.model import AdaIR; import torch; m = AdaIR(decoder=True); m.eval(); out = m(torch.randn(1,3,128,128)); print('OK', out.shape)"
```

### Key caveats

1. **No `requirements.txt` in the original repo** — only `env.yaml` (conda). The update script installs deps via pip.
2. **`options.py` runs `parser.parse_args()` at import time**, which means importing it outside of `train.py` will parse `sys.argv`. Avoid importing `options` directly in test scripts; use `test.py`'s own argparse instead.
3. **`scikit-video` (`skvideo`)** is required by `utils/val_utils.py` but easy to miss — it's included in the update script.
4. **No automated test suite** exists. Verification = successful model import + forward pass + image utility checks.
5. **Training datasets** must be manually downloaded from Google Drive links listed in `INSTALL.md` and placed under `data/Train/` and `data/test/`. Pre-trained checkpoints go in `ckpt/`.
6. For standard training/testing commands, see `README.md`.
