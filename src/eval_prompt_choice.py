import os
import json
from glob import glob

import numpy as np
import cv2
from sklearn.metrics import jaccard_score

# ============================================================
# CONFIG (EDIT HERE IF NEEDED)
# ============================================================

HM_DIR = "results_clipseg_drivable"   # prediction directory
GT_DIR = "results/gt"                 # ground-truth directory
AFF = "drivable"                      # single affordance
THR = 0.25                            # binarization threshold
VERBOSE = False

# ============================================================
# UTILS
# ============================================================

def binarize(hm, thr):
    return (hm >= thr).astype(np.uint8)


def compute_iou(gt, pred):
    gtf = gt.flatten()
    pr = pred.flatten()

    if gtf.sum() == 0 and pr.sum() == 0:
        return 1.0
    if gtf.sum() == 0:
        return 0.0

    return jaccard_score(gtf, pr, average="binary")


def find_basenames_from_dir(hm_dir):
    pattern = os.path.join(hm_dir, f"*_{AFF}_mask.npy")
    bases = set()

    for p in glob(pattern):
        base = os.path.basename(p).replace(f"_{AFF}_mask.npy", "")
        bases.add(base)

    return sorted(bases)


def resize_pred_to_gt(pred, gt_shape):
    if pred.ndim == 3:
        pred = pred.squeeze()

    pred = pred.astype(np.float32)
    h, w = gt_shape

    if pred.shape == (h, w):
        return pred

    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

    if pred.max() > 1 or pred.min() < 0:
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    return pred

# ============================================================
# EVALUATION
# ============================================================

def evaluate_all_images():

    imgs = find_basenames_from_dir(HM_DIR)

    if not imgs:
        print(f"[WARN] No drivable predictions found in {HM_DIR}")
        return {}

    scores = []

    for img in imgs:
        pred_path = os.path.join(HM_DIR, f"{img}_{AFF}_mask.npy")
        gt_path = os.path.join(GT_DIR, f"{img}_{AFF}.npy")

        if not os.path.exists(gt_path):
            if VERBOSE:
                print(f"[SKIP] Missing GT for {img}")
            continue

        pred = np.load(pred_path)
        gt = np.load(gt_path).astype(np.uint8)

        try:
            pred = resize_pred_to_gt(pred, gt.shape)
        except Exception as e:
            print(f"[ERROR] Resize failed for {img}: {e}")
            continue

        bin_pred = binarize(pred, THR)

        if bin_pred.shape != gt.shape:
            print(f"[WARN] Shape mismatch for {img}")
            continue

        iou = compute_iou(gt, bin_pred)
        scores.append(iou)

    mean_iou = float(np.mean(scores)) if scores else None
    std_iou = float(np.std(scores)) if scores else None

    summary = {
        AFF: (mean_iou, std_iou)
    }

    print("\n================ EVALUATION RESULT ================")
    print(f"Directory : {HM_DIR}")
    print(f"Threshold : {THR}")
    print(f"Affordance: {AFF}")
    print("--------------------------------------------------")
    print(summary)

    out_json = os.path.join(HM_DIR, "prompt_eval_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SAVED] {out_json}")
    print("==================================================\n")

    return summary

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_all_images()
