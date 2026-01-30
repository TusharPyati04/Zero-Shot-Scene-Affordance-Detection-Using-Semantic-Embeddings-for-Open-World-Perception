import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from prompt_list import PROMPTS

# ----------------------------
# USER SETTINGS
# ----------------------------
ADVERSE_IMG_DIR = "D:\\Tushar\\COLLEGE\\7th_SEM\\REU\\FDA\\Dataset\\adverse_images"
REFERENCE_IMG_DIR = "D:\\Tushar\\COLLEGE\\7th_SEM\\REU\\FDA\\Dataset\\reference_images"

OUT_DIR = "D:\\Tushar\\COLLEGE\\7th_SEM\\REU\\CLIPSEG_FDA_RESULTS"

THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FDA_BETA = 0.002

# ----------------------------
# CREATE OUTPUT FOLDERS
# ----------------------------
for sub in ["fda_images", "heatmaps", "masks", "overlays"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

# ----------------------------
# FDA FUNCTION
# ----------------------------
def apply_fda(source, target, beta=0.01):
    src_fft = np.fft.fft2(source, axes=(0, 1))
    tgt_fft = np.fft.fft2(target, axes=(0, 1))

    src_amp, src_phase = np.abs(src_fft), np.angle(src_fft)
    tgt_amp = np.abs(tgt_fft)

    h, w = src_amp.shape[:2]
    b = int(min(h, w) * beta)

    src_amp[:b, :b] = tgt_amp[:b, :b]
    src_amp[-b:, :b] = tgt_amp[-b:, :b]
    src_amp[:b, -b:] = tgt_amp[:b, -b:]
    src_amp[-b:, -b:] = tgt_amp[-b:, -b:]

    fft = src_amp * np.exp(1j * src_phase)
    img = np.fft.ifft2(fft, axes=(0, 1))

    return np.real(img)

# ----------------------------
# LOAD CLIPSEG
# ----------------------------
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined"
).to(DEVICE)

model.eval()

# ----------------------------
# LOAD REFERENCE IMAGE
# ----------------------------
ref_img_path = os.path.join(REFERENCE_IMG_DIR, os.listdir(REFERENCE_IMG_DIR)[0])
ref_img = cv2.imread(ref_img_path)
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

# ----------------------------
# PROCESS ALL IMAGES
# ----------------------------
image_list = sorted(os.listdir(ADVERSE_IMG_DIR))

for img_name in image_list:

    img_path = os.path.join(ADVERSE_IMG_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ref_resized = cv2.resize(ref_img, (img.shape[1], img.shape[0]))

    # ---------- FDA ----------
    img_fda = apply_fda(
        img.astype(np.float32),
        ref_resized.astype(np.float32),
        beta=FDA_BETA
    )

    img_fda = np.clip(img_fda, 0, 255).astype(np.uint8)

    cv2.imwrite(
        f"{OUT_DIR}/fda_images/{img_name}",
        cv2.cvtColor(img_fda, cv2.COLOR_RGB2BGR)
    )

    pil_img = Image.fromarray(img_fda)

    # ---------- DRIVABLE ----------
    prompt = PROMPTS["drivable"][0]

    inputs = processor(
        text=prompt,
        images=pil_img,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    heatmap = torch.sigmoid(outputs.logits)[0].cpu().numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    np.save(
        f"{OUT_DIR}/heatmaps/{img_name}_drivable.npy",
        heatmap
    )

    mask = (heatmap >= THRESHOLD).astype(np.uint8)

    cv2.imwrite(
        f"{OUT_DIR}/masks/{img_name}_drivable.png",
        mask * 255
    )

    overlay = img_fda.copy()
    overlay[mask == 1] = [255, 0, 0]

    blended = cv2.addWeighted(img_fda, 0.6, overlay, 0.4, 0)

    cv2.imwrite(
        f"{OUT_DIR}/overlays/{img_name}_drivable.png",
        cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    )

    print(f"[DONE] {img_name} - drivable (FDA)")

print("\nAll images processed successfully WITH FDA.")
