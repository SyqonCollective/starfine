# StarNet Fine-tuning

Advanced fine-tuning system for StarNet astronomical image processing.

## Quick Start

```bash
# Basic fine-tuning
python finetune/finetune_starnet_v2.py \
  --model_path ./finetune.pth \
  --epochs 60 \
  --batch_size 24 \
  --lr 2e-5

# Production fine-tuning
python finetune/finetune_starnet_v2.py \
  --model_path ./finetune.pth \
  --data_dir . \
  --lr 2e-5 \
  --epochs 60 \
  --batch_size 24 \
  --patience 12 \
  --output_dir ./outputs_v2
```

## Features

- ✅ Balanced Loss (L1 + SSIM + Edge + Perceptual)
- ✅ Attention-based U-Net architecture
- ✅ Astro-specific data augmentation
- ✅ EMA (Exponential Moving Average)
- ✅ Mixed precision training
- ✅ Cross-platform (CUDA/MPS/CPU)
- ✅ Early stopping & checkpointing

## Files

- `finetune/finetune_starnet_v2.py` - Main fine-tuning script
- `starnet_raw_output.py` - Inference GUI
- `unet_starnet.py` - Model architecture
