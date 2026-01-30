# src/clip_grid_baseline.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import math
import numpy as np
from PIL import Image
import cv2
import torch
import clip
from prompt_list import PROMPTS
import matplotlib.pyplot as plt

# ==== Configuration ====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID = 16                # 16x16 grid (low memory). Try 24 if comfortable.
PATCH_SIZE = 224         # CLIP image crop size
UPSAMPLE_TO = None       # None => upsample to original image size
OUT_DIR = "results"
# Set your example images directory (absolute or relative path)
EXAMPLES_DIR = r"D:\\Tushar\\COLLEGE\\7th_SEM\\REU\\Zero-shot\\data\\examples"
# ========================

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)

def extract_patch_features(img_pil, model, preprocess, grid=16):
    W, H = img_pil.size
    xs = np.linspace(0, W, grid+1, dtype=int)
    ys = np.linspace(0, H, grid+1, dtype=int)
    feats = []
    coords = []
    device = DEVICE
    model.eval()
    with torch.no_grad():
        for j in range(grid):
            row_feats = []
            for i in range(grid):
                x0, x1 = xs[i], xs[i+1]
                y0, y1 = ys[j], ys[j+1]
                crop = img_pil.crop((x0, y0, x1, y1)).resize((PATCH_SIZE, PATCH_SIZE))
                img_t = preprocess(crop).unsqueeze(0).to(device)
                img_feat = model.encode_image(img_t)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                row_feats.append(img_feat.cpu().numpy().squeeze())
                coords.append((x0, y0, x1, y1))
            feats.append(np.stack(row_feats))
    feats = np.stack(feats)  # grid x grid x D
    return feats, coords, (W, H)

def compute_text_features(prompts, model):
    device = DEVICE
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats.cpu().numpy()

def heatmap_from_feats(patch_feats, text_feat):
    Gx, Gy, D = patch_feats.shape
    pf = patch_feats.reshape(-1, D)
    pf = pf / np.linalg.norm(pf, axis=1, keepdims=True)
    tf = text_feat / np.linalg.norm(text_feat)
    sims = (pf @ tf).reshape(Gx, Gy)
    return sims

def normalize_heatmap(h):
    h = h - h.min()
    if h.max() > 1e-8:
        h = h / h.max()
    return h

def upsample_heatmap(h, size):
    return cv2.resize(h.astype(np.float32), (size[0], size[1]), interpolation=cv2.INTER_CUBIC)

def visualize_and_save(img_pil, heatmaps_dict, out_path):
    img_np = np.array(img_pil)
    H, W = img_np.shape[:2]
    n = len(heatmaps_dict)
    fig_h = 3
    fig, axs = plt.subplots(1, n + 1, figsize=(3 * (n + 1), fig_h))
    axs[0].imshow(img_np)
    axs[0].axis("off")
    axs[0].set_title("image")
    for ax, (k, hm) in zip(axs[1:], heatmaps_dict.items()):
        ax.imshow(img_np)
        ax.imshow(hm, alpha=0.5, cmap="jet")
        ax.axis("off")
        ax.set_title(k)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def run_inference(img_path, grid=GRID):
    print(f"\nProcessing: {img_path}")
    img_pil = Image.open(img_path).convert("RGB")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    patch_feats, coords, orig_size = extract_patch_features(img_pil, model, preprocess, grid=grid)
    heatmaps = {}
    for affordance, prompts in PROMPTS.items():
        tfeats = compute_text_features(prompts, model)
        accum = None
        for idx in range(tfeats.shape[0]):
            sims = heatmap_from_feats(patch_feats, tfeats[idx])
            if accum is None:
                accum = sims
            else:
                accum += sims
        avg = accum / tfeats.shape[0]
        avg = normalize_heatmap(avg)
        up = upsample_heatmap(avg, (orig_size[0], orig_size[1]))
        heatmaps[affordance] = up
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(OUT_DIR, f"{base}_heatmaps.png")
    visualize_and_save(img_pil, heatmaps, out_path)
    print("Saved:", out_path)
    for k, v in heatmaps.items():
        np.save(os.path.join(OUT_DIR, f"{base}_{k}_heatmap.npy"), v)
    return heatmaps

if __name__ == "__main__":
    print(f"\n>>> Using example image directory: {EXAMPLES_DIR}")
    import glob
    targets = glob.glob(os.path.join(EXAMPLES_DIR, "*.*"))
    if not targets:
        print("⚠️ No images found in the examples directory.")
    else:
        for t in targets:
            run_inference(t, grid=GRID)
