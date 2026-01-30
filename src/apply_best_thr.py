# src/apply_best_thr_and_clean.py
import os, glob, numpy as np, cv2
HM_DIR = "results_clipseg"
OUT_DIR = os.path.join(HM_DIR, "postprocessed")
os.makedirs(OUT_DIR, exist_ok=True)
THRS = {"drivable":0.60}

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

bases = sorted(set([os.path.basename(p).rsplit("_",2)[0] for p in glob.glob(os.path.join(HM_DIR,"*_mask.npy"))]))
for base in bases:
    for aff, thr in THRS.items():
        f = os.path.join(HM_DIR, f"{base}_{aff}_mask.npy")
        if not os.path.exists(f): continue
        arr = np.load(f).astype(float)
        if arr.max() > 1.0 or arr.min() < 0.0:
            arr = (arr - arr.min())/(arr.max()-arr.min()+1e-8)
        binm = (arr >= thr).astype(np.uint8)
        clean = clean_mask_bin(binm, min_area=200, ksize=7)
        np.save(os.path.join(OUT_DIR, f"{base}_{aff}_mask.npy"), clean)
        cv2.imwrite(os.path.join(OUT_DIR, f"{base}_{aff}_mask_clean.png"), (clean*255).astype('uint8'))
print("Saved cleaned masks to", OUT_DIR)
