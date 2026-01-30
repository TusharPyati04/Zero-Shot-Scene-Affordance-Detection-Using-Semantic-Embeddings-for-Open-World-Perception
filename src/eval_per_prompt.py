# src/eval_per_prompt.py
import os, numpy as np, json, cv2
from glob import glob
from sklearn.metrics import jaccard_score
from prompt_list_2 import PROMPTS

PER_DIR = "results_clipseg_perprompt"   # where your per-prompt .npy masks are
GT_DIR = "results/gt"
OUT_JSON = os.path.join(PER_DIR, "perprompt_iou_summary.json")
THR = 0.5   # threshold used to binarize predictions for per-prompt comparison (change if you want)

def compute_iou(gt, pred):
    gtf = gt.flatten(); pr = pred.flatten()
    # both empty -> perfect (keeps prior behaviour)
    if gtf.sum()==0 and pr.sum()==0: return 1.0
    if gtf.sum()==0: return 0.0
    return jaccard_score(gtf, pr, average='binary')

def load_np(path):
    try:
        return np.load(path)
    except Exception as e:
        print(f"[WARN] failed loading {path}: {e}")
        return None

# Build number of prompts from PROMPTS
num_prompts = {aff: len(prompts) for aff,prompts in PROMPTS.items()}

# find bases (images) by scanning files
files = glob(os.path.join(PER_DIR,"*_p*_mask.npy"))
bases = sorted(set([os.path.basename(p).rsplit("_",3)[0] for p in files]))
print("Found bases:", len(bases))

# Prepare result structure
results = {aff: {i: [] for i in range(num_prompts[aff])} for aff in PROMPTS.keys()}

for base in bases:
    for aff, n_prom in num_prompts.items():
        gt_path = os.path.join(GT_DIR, f"{base}_{aff}.npy")
        if not os.path.exists(gt_path):
            # no GT for this base-affordance
            #print(f"[SKIP] GT missing: {gt_path}")
            continue
        gt = load_np(gt_path)
        if gt is None:
            continue
        # ensure GT is binary 0/1 uint8
        gt = (gt > 0).astype(np.uint8)
        H_gt, W_gt = gt.shape[:2]

        for i in range(n_prom):
            pred_path = os.path.join(PER_DIR, f"{base}_{aff}_p{i}_mask.npy")
            if not os.path.exists(pred_path):
                #print(f"[SKIP] pred missing: {pred_path}")
                continue
            pred = load_np(pred_path)
            if pred is None:
                continue
            pred = np.array(pred, dtype=np.float32)

            # normalize pred to 0..1 safely
            pmin, pmax = float(pred.min()), float(pred.max())
            if pmax - pmin > 1e-8:
                pred = (pred - pmin) / (pmax - pmin)

            # if pred has different shape, resize to GT size
            if pred.shape[:2] != gt.shape[:2]:
                # cv2 resize expects (width, height)
                pred = cv2.resize(pred.astype(np.float32), (W_gt, H_gt), interpolation=cv2.INTER_CUBIC)

            # binarize with THR
            bin_pred = (pred >= THR).astype(np.uint8)

            # compute IoU
            try:
                iou = compute_iou(gt, bin_pred)
            except Exception as e:
                print(f"[ERROR] compute_iou failed for {base} {aff} p{i}: {e}")
                continue
            results[aff][i].append(float(iou))

# Summarize: mean, std, count
summary = {}
for aff, per in results.items():
    summary[aff] = {}
    for i, vals in per.items():
        if vals:
            summary[aff][f"p{i}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        else:
            summary[aff][f"p{i}"] = None

# print compact summary (best prompt per affordance)
print("\nPer-prompt IoU summary (means):")
best = {}
for aff, per in summary.items():
    best_aff = (None, -1.0)
    for p, stats in per.items():
        if stats is None: continue
        m = stats["mean"]
        print(f"  {aff} {p}: mean={m:.4f} std={stats['std']:.4f} n={stats['n']}")
        if m > best_aff[1]:
            best_aff = (p, m)
    best[aff] = best_aff

print("\nBest prompt per affordance (by mean IoU):")
for aff, (p, m) in best.items():
    print(f"  {aff}: {p} (mean IoU={m:.4f})")

# Save JSON
os.makedirs(PER_DIR, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump({"summary": summary, "best": best}, f, indent=2)
print("\nSaved results to", OUT_JSON)
