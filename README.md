# Vision-Transformer-ViT-on-CIFAR-10
Reimplementation of An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)

> A from-scratch PyTorch implementation of Vision Transformer for image classification — no external ViT libraries, just pure PyTorch.

**Dataset**: CIFAR-10 (10 classes, 32×32 images)  
**Model**: Mini-ViT built with PyTorch  
**Goal**: Understand ViT architecture and train it end-to-end  
**Best Accuracy**: ~72.57% after 20 epochs on a GPU

---

## Model Configuration

| Hyperparameter     | Value              |
|--------------------|--------------------|
| Image size         | 32 × 32            |
| Patch size         | 4 × 4              |
| Number of patches  | 64                 |
| Embedding dim      | 128                |
| Transformer depth  | 6 blocks           |
| Attention heads    | 4                  |
| MLP ratio          | 4×  (512 hidden)   |
| Dropout            | 0.1                |
| **Total params**   | **1,205,898**      |

---

## Results

### Training Curves

| Epoch | Train Loss | Test Accuracy |
|-------|-----------|---------------|
| 1     | 1.7838    | 42.33%        |
| 5     | 1.1867    | 60.66%        |
| 10    | 0.9702    | 67.41%        |
| 15    | 0.8447    | 71.54%        |
| 20    | 0.7866    | 72.53%        |

**Best Test Accuracy: 72.57%** (epoch 17–18)

### Per-Class Accuracy

| Class  | Accuracy | Class  | Accuracy |
|--------|----------|--------|----------|
| plane  | 75.9%    | dog    | 65.8%    |
| car    | 88.3%    | frog   | 82.4%    |
| bird   | 58.2%    | horse  | 80.6%    |
| cat    | 56.4%    | ship   | 84.2%    |
| deer   | 75.1%    | truck  | 80.0%    |

> **Hardest classes**: `cat` (56.4%) and `bird` (58.2%) — high intra-class variation.  
> **Easiest classes**: `car` (88.3%) and `ship` (84.2%) — distinctive shapes.

### ViT vs CNN Comparison

| Model              | Params  | Best Accuracy |
|--------------------|---------|---------------|
| ViT (Transformer)  | 1.21 M  | ~72.5%        |
| SmallCNN (ResNet)  | ~1.2 M  | ~80–85%       |

> CNNs still have an inductive bias advantage on small 32×32 images. ViTs typically need larger images or pre-training to match CNN performance.

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Kishanmvs/Vision-Transformer-ViT-on-CIFAR-10.git
cd Vision-Transformer-ViT-on-CIFAR-10
```

### 2. Install dependencies

```bash
pip install torch torchvision matplotlib seaborn scikit-learn Pillow requests
```

### 3. Run in Jupyter or Colab

```bash
jupyter notebook vit_cifar10_full.ipynb
```

> **Tip**: Use a GPU runtime in Colab (Runtime → Change runtime type → T4 GPU). Training takes ~9 min on GPU vs ~2 hrs on CPU.

---

## Key Findings

- **ViT needs more data**: Without pre-training, ViT underperforms CNNs on small datasets (CIFAR-10). Pre-training on ImageNet-21k first would push accuracy above 95%.
- **Low-resolution is hard**: 32×32 images give only 64 patches — very limited spatial information for attention to work with.
- **cat vs dog confusion**: The most common error is `cat → dog` and `dog → cat`, which makes intuitive sense.
- **Attention rollout works**: The attention maps clearly highlight salient object regions even without supervision.
- **Training is still converging** at epoch 20 — running for 50–100 epochs would improve accuracy further.

---

## Improvements & Next Steps

- [ ] **Longer training**: 50–100 epochs with warm-up schedule
- [ ] **Stronger augmentation**: AutoAugment / RandAugment / Mixup / CutMix
- [ ] **Larger model**: Increase `embed_dim=256`, `depth=8`, `num_heads=8`
- [ ] **Pre-training**: Load DeiT or ViT-S weights pre-trained on ImageNet
- [ ] **Patch size 2×2**: More patches (256) → richer spatial information
- [ ] **Label smoothing**: Add `label_smoothing=0.1` to CrossEntropyLoss

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.0.0
Pillow>=9.0.0
numpy>=1.21.0
requests>=2.28.0
```

---

## References

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [Attention Rollout (Abnar & Zuidema, 2020)](https://arxiv.org/abs/2005.00928)
- [Training data-efficient image transformers — DeiT (Touvron et al., 2021)](https://arxiv.org/abs/2012.12877)

---

## License

MIT License — feel free to use and modify.

---

*Built with PyTorch from scratch — no external ViT libraries.*
