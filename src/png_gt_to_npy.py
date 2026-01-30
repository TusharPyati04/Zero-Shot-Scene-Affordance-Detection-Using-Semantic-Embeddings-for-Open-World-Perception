# utils/png_gt_to_npy.py
from PIL import Image
import numpy as np
import os

PNG_GT_DIR = r"D:\Tushar\COLLEGE\7th_SEM\REU\Zero-shot\data\examples"
OUT = r"D:\Tushar\COLLEGE\7th_SEM\REU\Zero-shot\results\gt"
os.makedirs(OUT, exist_ok=True)

for f in os.listdir(PNG_GT_DIR):
    if not f.lower().endswith(".png"):
        continue
    # expects filenames like: <base>_drivable.png
    base, aff = f.rsplit("_", 1)
    aff = aff.replace(".png","")
    arr = np.array(Image.open(os.path.join(PNG_GT_DIR, f)).convert("L"))
    mask = (arr > 127).astype(np.uint8)
    np.save(os.path.join(OUT, f"{base}_{aff}.npy"), mask)

print("Converted PNG GT to .npy in:", OUT)
