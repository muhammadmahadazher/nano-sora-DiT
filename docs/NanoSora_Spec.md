# Project Spec: Nano-Sora (DiT + Flow Matching)

## 1. Objective
Build a "Nano" version of the Sora/Stable Diffusion 3 architecture. It must generate images (CIFAR-10 32x32) using a **Diffusion Transformer (DiT)** trained with **Flow Matching** (Rectified Flow).

## 2. Technical Stack
- **Language:** Python 3.10+
- **Framework:** PyTorch (CUDA enabled for RTX 4060)
- **Key Libraries:** `einops` (for tensor ops), `diffusers` (for VAE only), `wandb` (optional, for logging).
- **Hardware:** NVIDIA RTX 4060 (8GB VRAM). Code must use `mixed_precision="fp16"`.

## 3. Module Architecture
The project must follow this exact directory structure:

nano_sora/
├── data/                   # Dataset handling
│   └── loader.py           # Returns PyTorch WebDataset/DataLoader for CIFAR-10
├── models/
│   ├── dit.py              # The Core DiT Architecture (Layers, Blocks)
│   └── vae.py              # Wrapper for a tiny Autoencoder (optional for CIFAR, but good for scaling)
├── training/
│   ├── flow_matching.py    # The Rectified Flow Loss calculation
│   └── train.py            # Main training loop with checkpointing
├── inference.py            # Script to load model and generate images
├── config.py               # Hyperparameters (Batch size: 64, Learning Rate: 3e-4)
└── requirements.txt        # Dependencies

## 4. Detailed Implementation Requirements

### A. The DiT Block (`models/dit.py`)
Do not use a U-Net. Implement a Transformer with:
- **Patchify:** Convert 32x32 image into sequences of 2x2 or 4x4 patches.
- **Embeddings:** Standard Learnable Positional Embeddings.
- **DiT Block:** A standard Transformer block but with **Adaptive Layer Norm (adaLN)** that accepts `timestep` and `class_label` as conditioning.
  - `adaLN(x, c) = y_scale * Norm(x) + y_shift`
- **Final Layer:** Unpatchify back to image dimensions.

### B. Flow Matching Logic (`training/flow_matching.py`)
Use **Rectified Flow** (Straight line ODE).
- **Time sampling:** Uniform $t \in [0, 1]$.
- **Target:** The model predicts the velocity $v = (x_1 - x_0)$.
- **Loss:** MSE between `model_prediction` and `(image - noise)`.
- **Inference:** Use a simple Euler ODE solver (steps=50) to go from Noise ($t=0$) to Data ($t=1$).

### C. Training Loop (`training/train.py`)
- Use `torch.cuda.amp.GradScaler` for mixed precision (crucial for 8GB VRAM).
- Save checkpoints every 10 epochs.
- Log loss to console.