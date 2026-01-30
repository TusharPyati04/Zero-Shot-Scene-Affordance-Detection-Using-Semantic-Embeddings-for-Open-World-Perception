# src/clipseg_affordance.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from prompt_list import PROMPTS

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_DIR = r"D:\Tushar\COLLEGE\7th_SEM\REU\Zero-shot\data\examples"
OUT_DIR = "results_clipseg_drivable"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD CLIPSEG
# ============================================================

print("Loading CLIPSeg...")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined"
).to(DEVICE)
model.eval()

# ============================================================
# MAIN INFERENCE FUNCTION
# ============================================================

def run_clipseg(img_path):
    img = Image.open(img_path).convert("RGB")
    base = os.path.splitext(os.path.basename(img_path))[0]

    print(f"\nProcessing {base}")

    # --------------------------------------------------------
    # ONLY DRIVABLE AFFORDANCE
    # --------------------------------------------------------
    aff = "drivable"
    prompts = PROMPTS["drivable"]

    out_npy = os.path.join(OUT_DIR, f"{base}_{aff}_mask.npy")
    out_png = os.path.join(OUT_DIR, f"{base}_{aff}_mask.png")

    if os.path.exists(out_npy):
        print("  Skipping (already exists)")
        return

    all_masks = []

    # --------------------------------------------------------
    # PROMPT ENSEMBLE
    # --------------------------------------------------------
    for p in prompts:
        inputs = processor(
            text=[p],
            images=img,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            preds = model(**inputs)
            mask = torch.sigmoid(preds.logits).squeeze().cpu().numpy()
            all_masks.append(mask)

    if len(all_masks) == 0:
        print("  [WARN] No predictions")
        return

    # --------------------------------------------------------
    # STRONG RESPONSE AGGREGATION (BEST FOR DRIVABLE)
    # --------------------------------------------------------
    mask = np.percentile(all_masks, 85, axis=0)

    # Safe normalization
    if mask.max() > mask.min():
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    else:
        mask = np.zeros_like(mask)

    # --------------------------------------------------------
    # RESIZE TO ORIGINAL IMAGE SIZE
    # --------------------------------------------------------
    mask = np.array(
        Image.fromarray((mask * 255).astype(np.uint8))
        .resize(img.size, Image.BILINEAR)
    ) / 255.0

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------
    np.save(out_npy, mask)

    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5, cmap="jet")
    plt.axis("off")
    plt.title("drivable")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print("  Saved drivable mask")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    imgs = [
        os.path.join(IMG_DIR, f)
        for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ]

    for img_path in imgs:
        run_clipseg(img_path)
