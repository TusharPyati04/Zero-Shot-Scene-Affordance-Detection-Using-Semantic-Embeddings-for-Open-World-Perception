# src/ensemble_prompts_clipseg.py
import os, glob, numpy as np, cv2
from pathlib import Path

# Config
PERPROMPT_DIR = "results_clipseg_perprompt"    # produced by clipseg_per_prompt.py
AVG_DIR = os.path.join("results_clipseg", "ensemble_avg")
POST_DIR = os.path.join("results_clipseg", "ensemble_postproc")
os.makedirs(AVG_DIR, exist_ok=True)
os.makedirs(POST_DIR, exist_ok=True)

# Per-affordance thresholds discovered earlier (tune if you want)
THRS = {"drivable":0.60, "crossable":0.70, "obstructed":0.10}

# cleaning function (same as before)
def clean_mask_bin(mask_bin, min_area=300, ksize=7):
    m = (mask_bin*255).astype('uint8')
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8, cv2.CV_32S)
    out = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels==i] = 255
    return (out//255).astype(np.uint8)

# find bases by scanning per-prompt folder
all_files = glob.glob(os.path.join(PERPROMPT_DIR, "*_p*_mask.npy"))
bases = set()
for p in all_files:
    name = os.path.basename(p)
    # name format: <base>_<aff>_p<idx>_mask.npy
    # split off last 4 parts
    parts = name.rsplit("_", 4)
    if len(parts) >= 4:
        base = parts[0]
        bases.add(base)
bases = sorted(bases)
print("Found bases:", len(bases))

for base in bases:
    # for each affordance, collect all prompt masks
    for aff in THRS.keys():
        pattern = os.path.join(PERPROMPT_DIR, f"{base}_{aff}_p*_mask.npy")
        files = sorted(glob.glob(pattern))
        if len(files) == 0:
            continue
        # load & resize masks to same resolution (use first mask shape)
        masks = []
        ref = np.load(files[0])
        H, W = ref.shape
        for f in files:
            m = np.load(f)
            if m.shape != (H, W):
                # resize to HxW
                m = cv2.resize(m.astype(float), (W, H), interpolation=cv2.INTER_CUBIC)
            masks.append(m.astype(np.float32))
        avg = np.mean(np.stack(masks, axis=0), axis=0)  # HxW float
        # save averaged heatmap
        np.save(os.path.join(AVG_DIR, f"{base}_{aff}_avg_heatmap.npy"), avg)
        # save visualization png
        try:
            import matplotlib.pyplot as plt
            img_jpg = Path("data/examples") / f"{base}.jpg"
            img_png = Path("data/examples") / f"{base}.png"
            img_path = img_jpg if img_jpg.exists() else (img_png if img_png.exists() else None)
            if img_path:
                img = plt.imread(str(img_path))
                plt.imshow(img); plt.imshow(avg, alpha=0.5, cmap="jet"); plt.axis("off")
                plt.savefig(os.path.join(AVG_DIR, f"{base}_{aff}_avg.png"), dpi=150)
                plt.close()
        except Exception:
            pass
        # binarize using THRS and clean
        thr = THRS.get(aff, 0.4)
        binm = (avg >= thr).astype(np.uint8)
        clean = clean_mask_bin(binm, min_area=200, ksize=7)
        # Save cleaned mask in POST_DIR with evaluator-friendly name: <base>_<aff>_mask.npy
        np.save(os.path.join(POST_DIR, f"{base}_{aff}_mask.npy"), clean)
        cv2.imwrite(os.path.join(POST_DIR, f"{base}_{aff}_mask.png"), (clean*255).astype('uint8'))
print("Saved averaged heatmaps to", AVG_DIR)
print("Saved postprocessed binary masks to", POST_DIR)
