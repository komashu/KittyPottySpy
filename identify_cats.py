import os
import csv
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from sklearn.neighbors import NearestNeighbors

# --- Runtime/env tuning (quiet + sane CPU threads; optional) ---
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(".")
CATS_DIR = PROJECT_ROOT / "cats"                 # gallery: cats/<Name>/*.jpg
CROPS_DIR = PROJECT_ROOT / "crops"               # detection crops
FRAMES_DIR = PROJECT_ROOT / "frames"             # annotated frames (boxes only)
META_DIR = PROJECT_ROOT / "meta"
DETECTIONS_CSV_PATH = META_DIR / "detections.csv"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"   # output buckets of crops
PREDICTIONS_DIR.mkdir(exist_ok=True)
FRAMES_LABELED_DIR = PROJECT_ROOT / "frames_labeled"  # frames with cat name overlay
FRAMES_LABELED_DIR.mkdir(exist_ok=True)

# ----------------------------
# Config
# ----------------------------
EMBED_IMAGE_SIZE = 224
EMBED_BATCH_SIZE = 32
# cosine *distance* threshold (lower = more similar). > threshold -> "unknown"
UNKNOWN_DISTANCE_THRESHOLD = 0.28

# ----------------------------
# Model: ResNet18 -> 512-D embeddings
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_EXTRACTOR = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
FEATURE_EXTRACTOR.fc = nn.Identity()
FEATURE_EXTRACTOR.eval().to(DEVICE)

TRANSFORM = T.Compose([
    T.Resize((EMBED_IMAGE_SIZE, EMBED_IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def load_image_paths(folder: Path,
                     extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")) -> List[Path]:
    """Return sorted list of image file paths under `folder` with given extensions."""
    return sorted([path for path in folder.rglob("*")
                   if path.suffix.lower() in extensions])


def embed_images(image_paths: List[Path]) -> np.ndarray:
    """Compute 512-D embeddings for a list of images using the global FEATURE_EXTRACTOR."""
    embeddings: List[np.ndarray] = []
    batch_tensors: List[torch.Tensor] = []
    batch_paths: List[Path] = []

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Embedding", unit="img"):
            try:
                pil_image = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            batch_tensors.append(TRANSFORM(pil_image))
            batch_paths.append(image_path)

            if len(batch_tensors) == EMBED_BATCH_SIZE:
                batch_tensor = torch.stack(batch_tensors).to(DEVICE)
                batch_features = FEATURE_EXTRACTOR(batch_tensor).cpu().numpy()
                embeddings.append(batch_features)
                batch_tensors, batch_paths = [], []

        # flush remaining
        if batch_tensors:
            batch_tensor = torch.stack(batch_tensors).to(DEVICE)
            batch_features = FEATURE_EXTRACTOR(batch_tensor).cpu().numpy()
            embeddings.append(batch_features)

    if not embeddings:
        return np.zeros((0, 512), dtype=np.float32)
    return np.vstack(embeddings).astype(np.float32)


def build_gallery() -> Tuple[np.ndarray, List[str], List[Path]]:
    """Load gallery images from cats/<Name> folders and return (embeddings, labels, paths)."""
    gallery_paths: List[Path] = []
    gallery_labels: List[str] = []

    for cat_folder in sorted(CATS_DIR.iterdir()):
        if not cat_folder.is_dir():
            continue
        label = cat_folder.name
        images_in_folder = load_image_paths(cat_folder)
        for image_path in images_in_folder:
            gallery_paths.append(image_path)
            gallery_labels.append(label)

    if not gallery_paths:
        raise SystemExit("No gallery images found in 'cats/*'. Add some cat photos first.")

    print(f"Found {len(gallery_paths)} gallery images across {len(set(gallery_labels))} cats.")
    gallery_embeddings = embed_images(gallery_paths)
    return gallery_embeddings, gallery_labels, gallery_paths


def nearest_neighbor_classifier(gallery_embeddings: np.ndarray,
                                gallery_labels: List[str]):
    """Build a k-NN predictor over gallery embeddings (cosine distance)."""
    neighbor_index = NearestNeighbors(n_neighbors=3, metric="cosine")
    neighbor_index.fit(gallery_embeddings)

    def predict_label(embedding: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        distances, indices = neighbor_index.kneighbors(
            embedding.reshape(1, -1), return_distance=True
        )
        distances = distances[0]
        indices = indices[0]

        label_votes: Dict[str, List[float]] = {}
        for distance_value, neighbor_idx in zip(distances, indices):
            neighbor_label = gallery_labels[neighbor_idx]
            label_votes.setdefault(neighbor_label, []).append(float(distance_value))

        # average distance per label; choose the smallest
        label_distance_pairs = [(label, float(np.mean(d_list)))
                                for label, d_list in label_votes.items()]
        label_distance_pairs.sort(key=lambda pair: pair[1])
        best_label, best_distance = label_distance_pairs[0]

        # also return raw top-3 neighbors (label, distance)
        raw_neighbors = [(gallery_labels[idx], float(distances[j]))
                         for j, idx in enumerate(indices)]
        return best_label, best_distance, raw_neighbors

    return predict_label


def copy_crop_into_bucket(source_path: Path, predicted_label: str, distance_value: float) -> None:
    """Copy crop into predictions/<label>/ (or predictions/unknown/ if over threshold)."""
    if distance_value > UNKNOWN_DISTANCE_THRESHOLD:
        destination_dir = PREDICTIONS_DIR / "unknown"
    else:
        destination_dir = PREDICTIONS_DIR / predicted_label
    destination_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source_path, destination_dir / source_path.name)
    except Exception:
        # keep running even if an occasional copy fails
        pass


def load_detections() -> Dict[str, List[dict]]:
    """Map frame filename -> list of detection dicts with bbox and crop_file."""
    if not DETECTIONS_CSV_PATH.exists():
        print(f"Warning: {DETECTIONS_CSV_PATH} not found. Skipping frame labeling.")
        return {}

    detections_by_frame: Dict[str, List[dict]] = {}
    with open(DETECTIONS_CSV_PATH, newline="") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row.get("label") != "cat":
                continue
            frame_filename = row["frame_file"]
            detection_dict = {
                "det_idx": int(row["det_idx"]),
                "conf": float(row["conf"]),
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
                "crop_file": row["crop_file"],
                "pred": None,        # will fill in later
                "pred_dist": None,   # will fill in later
            }
            detections_by_frame.setdefault(frame_filename, []).append(detection_dict)

    return detections_by_frame


def draw_label_on_frame(frame_image, label_text, left_x, top_y):
    """Draw white text label above (x, y) on the frame."""
    cv2.putText(
        frame_image,
        label_text,
        (left_x, max(0, top_y - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )


def main():
    # 1) Build gallery and predictor
    gallery_embeddings, gallery_labels, _ = build_gallery()
    predict_label = nearest_neighbor_classifier(gallery_embeddings, gallery_labels)

    # 2) Embed crops
    crop_image_paths = load_image_paths(CROPS_DIR)
    if not crop_image_paths:
        print("No crops found in 'crops/'. Run the detector first.")
        return
    print(f"Found {len(crop_image_paths)} crop images to classify.")
    crop_embeddings = embed_images(crop_image_paths)

    # 3) Classify crops and copy into prediction buckets
    prediction_rows: List[Dict[str, str]] = []
    crop_name_to_prediction: Dict[str, Tuple[str, float]] = {}

    for crop_path, crop_embedding in zip(crop_image_paths, crop_embeddings):
        predicted_label, distance_value, neighbor_list = predict_label(crop_embedding)
        final_label = "unknown" if distance_value > UNKNOWN_DISTANCE_THRESHOLD else predicted_label

        crop_name_to_prediction[crop_path.name] = (final_label, distance_value)
        copy_crop_into_bucket(crop_path, predicted_label, distance_value)

        prediction_rows.append({
            "crop_path": str(crop_path),
            "predicted": final_label,
            "cosine_distance": f"{distance_value:.4f}",
            "neighbors": "; ".join([f"{label}:{dist:.3f}" for label, dist in neighbor_list]),
        })

    # 4) Save predictions CSV
    predictions_csv_path = PREDICTIONS_DIR / "predictions.csv"
    with open(predictions_csv_path, "w", newline="") as csv_file:
        fieldnames = ["crop_path", "predicted", "cosine_distance", "neighbors"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(prediction_rows)
    print(f"✅ Predictions CSV → {predictions_csv_path}")

    # 5) Overlay predicted names on frames (using detections CSV for box positions)
    detections_by_frame = load_detections()
    if not detections_by_frame:
        print("Skipping frame labeling (no detections metadata).")
        return

    for frame_filename, detection_list in detections_by_frame.items():
        frame_path = FRAMES_DIR / frame_filename
        if not frame_path.exists():
            continue

        frame_image = cv2.imread(str(frame_path))
        if frame_image is None:
            continue

        for detection in detection_list:
            crop_filename = detection["crop_file"]
            predicted_name, distance_value = crop_name_to_prediction.get(
                crop_filename, ("unknown", 9.99)
            )
            detection["pred"] = predicted_name
            detection["pred_dist"] = distance_value

            x_min, y_min, x_max, y_max = (
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"],
            )

            # color by certainty (unknown = gray, otherwise green)
            box_color = (120, 120, 120) if predicted_name == "unknown" else (0, 200, 0)
            cv2.rectangle(frame_image, (x_min, y_min), (x_max, y_max), box_color, 2)
            label_text = (
                f"{predicted_name} ({distance_value:.2f})"
                if predicted_name != "unknown"
                else "unknown"
            )
            draw_label_on_frame(frame_image, label_text, x_min, y_min)

        labeled_output_path = FRAMES_LABELED_DIR / frame_filename
        cv2.imwrite(str(labeled_output_path), frame_image)

    print(f"🖼️ Labeled frames written to → {FRAMES_LABELED_DIR}")


if __name__ == "__main__":
    main()