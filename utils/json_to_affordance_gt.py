# utils/json_to_affordance_gt.py
import os
import json
import numpy as np
from PIL import Image, ImageDraw
from glob import glob

# ---------------- USER CONFIG ----------------
JSON_DIR = r"D:\Tushar\COLLEGE\7th_SEM\REU\Zero-shot\data\dataset\labels"    # folder containing one-or-more JSON files like the example you pasted
IMG_DIR  = r"D:\Tushar\COLLEGE\7th_SEM\REU\Zero-shot\data\dataset\examples"       # folder with images named like "<name>.jpg" or "<name>.png"
OUT_GT   = r"results/gt"          # output folder (will be created)
os.makedirs(OUT_GT, exist_ok=True)

# Mapping from JSON 'category' strings (lowercased) to affordances.
# Edit these sets if your categories use different names.
DRIVABLE_CATS = {
    "area/drivable", "area/alternative", "drivable", "road", "paved_area"
}
CROSSABLE_CATS = {
    "crosswalk", "zebra", "sidewalk", "pedestrian_crossing", "curb"
}
OBSTRUCTED_CATS = {
    "car", "truck", "bus", "van", "bicycle", "motorcycle", "motorbike", "person",
    "rider", "vehicle", "traffic sign", "traffic light", "truck", "bus"
}
# ----------------------------------------------

def find_image_for_base(base_name):
    # try common extensions
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = os.path.join(IMG_DIR, base_name + ext)
        if os.path.exists(p):
            return p
    # try any file that startswith base (fallback)
    for p in glob(os.path.join(IMG_DIR, base_name + "*")):
        if os.path.isfile(p):
            return p
    return None

def rasterize_poly2d(poly2d, img_size):
    # poly2d is a list of lists: [ [x,y,"L"|"C"], [x,y,"C"], ... ]
    # convert to a simple list of (x,y) and draw polygon
    if not poly2d:
        return np.zeros((img_size[1], img_size[0]), dtype=np.uint8)  # HxW
    pts = []
    for p in poly2d:
        # some entries may be [x,y,"L"] or a two-element list
        if isinstance(p, (list, tuple)):
            if len(p) >= 2:
                x = float(p[0]); y = float(p[1])
                pts.append((x, y))
    if len(pts) < 3:
        # not a polygon; return empty
        return np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    W, H = img_size
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(pts, outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)

def rasterize_box2d(box2d, img_size):
    x1 = float(box2d.get("x1", 0))
    y1 = float(box2d.get("y1", 0))
    x2 = float(box2d.get("x2", 0))
    y2 = float(box2d.get("y2", 0))
    W, H = img_size
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.rectangle([x1, y1, x2, y2], outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)

def process_single_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base = data.get("name", None)
    if not base:
        # try filename
        base = os.path.splitext(os.path.basename(path))[0]
    # If image file exists, get size; otherwise try to infer from JSON (frames might contain imageSize)
    img_path = find_image_for_base(base)
    if img_path:
        with Image.open(img_path) as im:
            W, H = im.size
        print(f"[INFO] Found image for {base}: {img_path} -> size {(W,H)}")
    else:
        # fallback to searching for width/height in JSON frames (sometimes present)
        W = None; H = None
        frames = data.get("frames", [])
        if frames:
            for fr in frames:
                # sometimes frames have 'width'/'height' or objects have normalized coords - but in your example we have pixel coords
                pass
        if W is None:
            raise FileNotFoundError(f"Image for base '{base}' not found in {IMG_DIR}. Please place the image or change IMG_DIR.")
    img_size = (W, H)

    # initialize empty masks
    drivable_mask = np.zeros((H, W), dtype=np.uint8)
    crossable_mask = np.zeros((H, W), dtype=np.uint8)
    obstructed_mask = np.zeros((H, W), dtype=np.uint8)

    frames = data.get("frames", [])
    if not frames:
        print(f"[WARN] no frames in {path}; skipping")
        return

    # we'll process the first frame (typical for single-image JSON)
    frame = frames[0]
    objects = frame.get("objects", [])
    for obj in objects:
        cat = str(obj.get("category", "")).lower().strip()
        # polygon (poly2d) preferred, else box2d
        poly2d = obj.get("poly2d", None)
        box2d = obj.get("box2d", None)
        mask = None
        if poly2d:
            # poly2d may be a list-of-lists of coordinates or a list with nested yeh
            # in some cases poly2d is [[ [x,y,"L"], ... ]] (list containing list). Normalize:
            if isinstance(poly2d, list) and len(poly2d)>0 and isinstance(poly2d[0][0], (list, tuple)):
                # poly2d is [ polygon1, polygon2, ... ] where each polygon is list of points
                # rasterize all polygons and OR them
                accum = np.zeros((H, W), dtype=np.uint8)
                for poly in poly2d:
                    m = rasterize_poly2d(poly, img_size)
                    accum = np.logical_or(accum, m)
                mask = accum.astype(np.uint8)
            else:
                # poly2d is a single polygon: list of [x,y,...]
                mask = rasterize_poly2d(poly2d, img_size)
        elif box2d:
            mask = rasterize_box2d(box2d, img_size)
        else:
            # unknown shape - skip
            continue

        # decide which affordance(s) this category contributes to
        if cat in DRIVABLE_CATS or any(k in cat for k in ("area/drivable","drivable","lane")):
            drivable_mask = np.logical_or(drivable_mask, mask)
        if cat in CROSSABLE_CATS or "cross" in cat or "sidewalk" in cat:
            crossable_mask = np.logical_or(crossable_mask, mask)
        if cat in OBSTRUCTED_CATS or any(k in cat for k in ("car","truck","bus","person","vehicle","traffic sign","traffic light")):
            obstructed_mask = np.logical_or(obstructed_mask, mask)

    # convert bool -> uint8
    drivable_mask = drivable_mask.astype(np.uint8)
    crossable_mask = crossable_mask.astype(np.uint8)
    obstructed_mask = obstructed_mask.astype(np.uint8)

    # Save npy files
    np.save(os.path.join(OUT_GT, f"{base}_drivable.npy"), drivable_mask)
    np.save(os.path.join(OUT_GT, f"{base}_crossable.npy"), crossable_mask)
    np.save(os.path.join(OUT_GT, f"{base}_obstructed.npy"), obstructed_mask)
    print(f"[SAVED] {base} -> drivable/crossable/obstructed masks")

def main():
    # process all JSONs in JSON_DIR
    json_files = []
    # accept either a directory of per-image JSONs, or a single COCO-like json file
    if os.path.isdir(JSON_DIR):
        for p in glob(os.path.join(JSON_DIR, "*.json")):
            json_files.append(p)
    else:
        json_files = [JSON_DIR]

    if not json_files:
        print("No JSON files found in", JSON_DIR)
        return

    for jf in json_files:
        print("Processing", jf)
        try:
            process_single_json(jf)
        except Exception as e:
            print("Error processing", jf, ":", e)

if __name__ == "__main__":
    main()
