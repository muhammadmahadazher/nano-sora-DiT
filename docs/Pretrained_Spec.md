# Project Spec: Nano-Sora (Pre-trained Pivot)

## 1. Objective
Switch backend from custom training to the official pre-trained `facebook/DiT-XL-2-256` model using the Hugging Face `diffusers` library.
Upgrade the UI to a modern "Glassmorphism" aesthetic.

## 2. Technical Stack
- **Library:** `diffusers`, `transformers`, `accelerate`
- **Model:** `facebook/DiT-XL-2-256` (Class-Conditional ImageNet)
- **Hardware:** RTX 4060 (Use `torch.float16` for memory efficiency)

## 3. Implementation Requirements

### A. Dependencies (`requirements.txt`)
Add: `diffusers`, `transformers`, `accelerate`, `scipy`

### B. Inference Engine (`inference.py`)
- Import `DiTPipeline` from `diffusers`.
- Function `load_pipeline()`:
  - Load "facebook/DiT-XL-2-256".
  - Move to CUDA.
  - Enable `torch.float16`.
  - Use `DPMSolverMultistepScheduler` for fast inference (20 steps instead of 50).

### C. The "Beautiful" App (`app.py`)
- **Mapping:** Map the user's simple selection to ImageNet Class IDs:
  - "Tench" -> 0
  - "Goldfish" -> 1
  - "Corgi" -> 263
  - "Airplane" -> 404
  - "Tiger" -> 292
- **UI Design (CSS):**
  - Inject Custom CSS via `st.markdown`.
  - Background: Dark Gradient (#0e1117).
  - Cards: Semi-transparent white with blur (`backdrop-filter: blur(10px)`).
  - Buttons: Gradient color with hover effects.
- **Logic:**
  - Load pipeline (cached).
  - On "Generate": Run pipe(class_labels=[id]).
  - Display result in the center with a "Download" button.