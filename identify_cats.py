import os
import csv
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from sklearn.neighbors import NearestNeighbors

# ----------------------------
# Paths
# ----------------------------
ROOT = Path(".")
CATS_DIR = ROOT / "cats"           # gallery: cats/<Name>/*.jpg
CROPS_DIR = ROOT / "crops"         # detection crops
FRAMES_DIR = ROOT / "frames"       # annotated frames (boxes only)
META_DIR = ROOT / "meta"
DET_CSV = META_DIR / "detections.csv"
PRED_DIR = ROOT / "predictions"    # output buckets of crops
PRED_DIR.mkdir(exist_ok=True)
FRAMES_LABELED = ROOT / "frames_labeled"  # frames with cat name overlay
FRAMES_LABELED.mkdir(exist_ok=True)

# ----------------------------
# Config
# ----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
UNKNOWN_THRESHOLD = 0.28  # cosine distance threshold (lower = more similar)

# ----------------------------
# Model: ResNet18 -> 512-D embeddings
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()
resnet.eval().to(device)

tx = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def load_image_paths(folder: Path, exts=(".jpg", ".jpeg", ".png", ".webp")) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])

def embed_images(paths: List[Path]) -> np.ndarray:
    embs = []
    kept = []
    with torch.no_grad():
        batch = []
        batch_paths = []
        for p in tqdm(paths, desc="Embedding", unit="img"):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            batch.append(tx(img))
            batch_paths.append(p)
            if len(batch) == BATCH_SIZE:
                x = torch.stack(batch).to(device)
                feat = resnet(x).cpu().numpy()
                embs.append(feat)
                kept.extend(batch_paths)
                batch, batch_paths = [], []
        if batch:
            x = torch.stack(batch).to(device)
            feat = resnet(x).cpu().numpy()
            embs.append(feat)
            kept.extend(batch_paths)
    if not embs:
        return np.zeros((0, 512), dtype=np.float32)
    return np.vstack(embs).astype(np.float32)

def build_gallery() -> Tuple[np.ndarray, List[str], List[Path]]:
    all_paths, all_labels = [], []
    for cat_dir in sorted(CATS_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        label = cat_dir.name
        imgs = load_image_paths(cat_dir)
        for p in imgs:
            all_paths.append(p)
            all_labels.append(label)
    if not all_paths:
        raise SystemExit("No gallery images found in 'cats/*'. Add some cat photos first.")
    print(f"Found {len(all_paths)} gallery images across {len(set(all_labels))} cats.")
    embs = embed_images(all_paths)
    return embs, all_labels, all_paths

def nearest_neighbor_classifier(gal_embs: np.ndarray, gal_labels: List[str]):
    nn = NearestNeighbors(n_neighbors=3, metric="cosine")
    nn.fit(gal_embs)
    def predict(emb: np.ndarray) -> Tuple[str, float, List[Tuple[str,float]]]:
        dist, idx = nn.kneighbors(emb.reshape(1, -1), return_distance=True)
        dist, idx = dist[0], idx[0]
        votes = {}
        for d, i in zip(dist, idx):
            lab = gal_labels[i]
            votes.setdefault(lab, []).append(d)
        label_scores = [(lab, float(np.mean(ds))) for lab, ds in votes.items()]
        label_scores.sort(key=lambda x: x[1])
        best_label, best_dist = label_scores[0]
        raw_neighbors = [(gal_labels[i], float(dist[j])) for j, i in enumerate(idx)]
        return best_label, best_dist, raw_neighbors
    return predict

def bucket_copy(src: Path, cat_label: str, conf_dist: float):
    if conf_dist > UNKNOWN_THRESHOLD:
        out_dir = PRED_DIR / "unknown"
    else:
        out_dir = PRED_DIR / cat_label
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, out_dir / src.name)
    except Exception:
        pass

def load_detections() -> Dict[str, List[dict]]:
    """Map frame filename -> list of detections with bbox and crop_file."""
    if not DET_CSV.exists():
        print(f"Warning: {DET_CSV} not found. Skipping frame labeling.")
        return {}
    det_map: Dict[str, List[dict]] = {}
    with open(DET_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label") != "cat":
                continue
            frame = row["frame_file"]
            det = {
                "det_idx": int(row["det_idx"]),
                "conf": float(row["conf"]),
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
                "crop_file": row["crop_file"],
                "pred": None,          # will fill in later
                "pred_dist": None,     # will fill in later
            }
            det_map.setdefault(frame, []).append(det)
    return det_map

def draw_label(img, text, x, y):
    cv2.putText(img, text, (x, max(0, y-12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, lineType=cv2.LINE_AA)

def main():
    # 1) Build gallery NN
    gal_embs, gal_labels, _ = build_gallery()
    predict = nearest_neighbor_classifier(gal_embs, gal_labels)

    # 2) Embed crops
    crop_paths = load_image_paths(CROPS_DIR)
    if not crop_paths:
        print("No crops found in 'crops/'. Run the detector first.")
        return
    print(f"Found {len(crop_paths)} crop images to classify.")
    crop_embs = embed_images(crop_paths)

    # 3) Classify crops + bucket copies
    rows = []
    crop_to_pred: Dict[str, Tuple[str, float]] = {}
    for p, emb in zip(crop_paths, crop_embs):
        label, dist, neighbors = predict(emb)
        pred_label = "unknown" if dist > UNKNOWN_THRESHOLD else label
        crop_to_pred[p.name] = (pred_label, dist)
        bucket_copy(p, label, dist)
        rows.append({
            "crop_path": str(p),
            "predicted": pred_label,
            "cosine_distance": f"{dist:.4f}",
            "neighbors": "; ".join([f"{lab}:{d:.3f}" for lab, d in neighbors])
        })

    # 4) Save CSV
    csv_path = PRED_DIR / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "predicted", "cosine_distance", "neighbors"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Predictions CSV → {csv_path}")

    # 5) Overlay predicted names on frames (using detections CSV for box positions)
    det_map = load_detections()
    if not det_map:
        print("Skipping frame labeling (no detections metadata).")
        return

    for frame_file, dets in det_map.items():
        frame_path = FRAMES_DIR / frame_file
        if not frame_path.exists():
            continue
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        for d in dets:
            crop_name = d["crop_file"]
            pred, dist = crop_to_pred.get(crop_name, ("unknown", 9.99))
            d["pred"] = pred
            d["pred_dist"] = dist
            x1,y1,x2,y2 = d["x1"], d["y1"], d["x2"], d["y2"]

            # color by certainty (simple: unknown = gray, else green)
            color = (120,120,120) if pred == "unknown" else (0,200,0)
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            label_text = f"{pred} ({dist:.2f})" if pred != "unknown" else "unknown"
            draw_label(img, label_text, x1, y1)

        out_path = FRAMES_LABELED / frame_file
        cv2.imwrite(str(out_path), img)

    print(f"🖼️ Labeled frames written to → {FRAMES_LABELED}")

if __name__ == "__main__":
    main()
