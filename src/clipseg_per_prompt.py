import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from PIL import Image
import numpy as np
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from prompt_list import PROMPTS

# Config
IMG_DIR = r"D:\\Tushar\\COLLEGE\\7th_SEM\\REU\\Zero-shot\\data\\examples"  # change if needed
OUT_DIR = "results_clipseg_perprompt"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIPSeg model...")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE)
model.eval()

def save_mask(base, aff, idx, mask, out_dir=OUT_DIR):
    # mask assumed float (0..1) or arbitrary -> normalize then save
    m = np.array(mask, dtype=np.float32)
    if m.max() > 1.0 or m.min() < 0.0:
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    np.save(os.path.join(out_dir, f"{base}_{aff}_p{idx}_mask.npy"), m)
    # Also save PNG overlay for quick inspection
    try:
        import matplotlib.pyplot as plt
        img_path = os.path.join(IMG_DIR, f"{base}.jpg")
        if not os.path.exists(img_path):
            # try png
            img_path = os.path.join(IMG_DIR, f"{base}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img); plt.imshow(m, alpha=0.5, cmap="jet"); plt.axis("off")
            plt.savefig(os.path.join(out_dir, f"{base}_{aff}_p{idx}_mask.png"), dpi=150)
            plt.close()
    except Exception:
        pass

def run():
    imgs = [p for p in os.listdir(IMG_DIR) if p.lower().endswith(('.jpg','.png'))]
    imgs = sorted(imgs)
    if not imgs:
        print("No images found in", IMG_DIR); return
    for imname in imgs:
        base = Path(imname).stem
        print("Processing", base)
        img = Image.open(os.path.join(IMG_DIR, imname)).convert("RGB")
        for aff, prompts in PROMPTS.items():
            for idx, prompt in enumerate(prompts):
                text = prompt
                inputs = processor(text=[text], images=img, padding=True, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    preds = model(**inputs)
                    logits = preds.logits.squeeze().cpu().numpy()  # H x W
                    mask = 1.0 / (1.0 + np.exp(-logits))  # sigmoid if not already
                    # normalize
                    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
                save_mask(base, aff, idx, mask)
    print("Saved per-prompt masks into", OUT_DIR)

if __name__ == "__main__":
    run()
