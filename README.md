# AIRL-ASSIGNMENT

# Vision Transformer (ViT) – CIFAR-10 Classification

This project implements a **Vision Transformer (ViT)** from scratch for CIFAR-10 classification.  
The model is trained end-to-end without external pretraining, focusing on understanding patch embeddings, transformer blocks, and training methods.  

---

## Running on Google Colab
Both notebooks (`q1.ipynb`, `q2.ipynb`) are designed to run **top-to-bottom** on Colab without modification.  

To run:
1. Upload the notebook to Colab.
2. Select **GPU runtime** (`Runtime > Change runtime type > GPU`).
3. Execute all cells in order (`Runtime > Run all`).

---

## Best Model Configuration
The best-performing model used the following setup:

- **Image size:** 128×128  
- **Patch size:** 16×16 (better GPU efficiency)  
- **Embedding dimension:** 192  
- **Transformer depth:** 8 layers  
- **Attention heads:** 6  
- **MLP hidden dim:** 768  
- **Dropout:** 0.15  
- **Optimizer:** AdamW (`lr = 5e-4`, weight decay = 0.05)  
- **Scheduler:** Cosine decay with 5 warmup epochs  
- **Label smoothing:** 0.1  
- **Augmentations:** RandomCrop, HorizontalFlip, AutoAugment (CIFAR-10 policy), RandomErasing  
- **Epochs:** 50 (with early stopping)  
- **Batch size:** 128  

---

## Result
| Model              | Dataset   | Test Accuracy |
|--------------------|-----------|---------------|
| ViT (from scratch) | CIFAR-10  | **84%** |

---

## 🔍 Comparison with Fine-tuning
Training ViT **from scratch** is limited by dataset size (~50K images).  
When pretrained on large datasets and fine-tuned, ViTs achieve near-perfect accuracy:

- **Scratch Training:** ~84%  
- **ImageNet Pretraining:** ~98–99%  
- **JFT-300M Pretraining:** ~99%+  

## Analysis
- CIFAR-10 resized to **128×128** (instead of 224×224) for faster training
- **Patch size 16×16:** balanced GPU utilization & accuracy (smaller patches = better detail but slower).  
- **Depth/Width trade-off:** 8 layers, 192-dim chosen for higher capacity without OOM.  
- **Augmentation (AutoAugment, Cutout):** boosted generalization by +3–4%.  
- **Optimizer + Cosine schedule + warmup:** stabilized training and improved convergence.  
- **Label smoothing:** reduced overconfidence, improving test accuracy.  
- **Epochs (50):** enough for convergence on CIFAR-10; longer runs did not give major gains.
- **Early stopping** prevents over-training and saves compute
- Tested **Shifted Patch Tokenization (SPT)** to inject local inductive bias.
---
