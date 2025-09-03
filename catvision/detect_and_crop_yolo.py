#!/usr/bin/env python3
"""
Detect cats in newest frames under samples/ and write:
- annotated frames to frames/
- crops to crops/
- rows to meta/detections.csv

Features:
- Skips frames that already appear in detections.csv
- Processes only the newest MAX_FRAMES_PER_RUN frames (env)
- Progress bar with tqdm
- Optional ROI mask (meta/roi.json) to restrict detection area
- Auto day/night confidence selection (env or auto)
- Skips tiny/incomplete JPEGs
- Flush + fsync after each CSV row to avoid partial writes
- Graceful SIGTERM handling (Docker stop)
"""

# [001] -------- Imports --------
import csv
import json
import os
import signal
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# [020] -------- Quiet logs + sane CPU threads --------
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "3")))

# [030] -------- Paths --------
SAMPLES_DIR = Path("samples")
FRAMES_DIR = Path("frames")
CROPS_DIR = Path("crops")
META_DIR = Path("../meta")

META_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
CROPS_DIR.mkdir(exist_ok=True)

DETECTIONS_CSV_PATH = META_DIR / "detections.csv"
ROI_JSON_PATH = META_DIR / "roi.json"

# [045] -------- Config via env (CPU/AVX friendly defaults) --------
MODEL_NAME = os.getenv("DETECT_MODEL", "yolov8n.pt")       # nano model by default (fastest CPU)
IMAGE_SIZE = int(os.getenv("DETECT_IMGSZ", "512"))         # 512 is snappy on AVX-only CPUs
CONFIDENCE_THRESHOLD_DEFAULT = float(os.getenv("DETECT_CONF", "0.20"))
IOU_THRESHOLD = float(os.getenv("DETECT_IOU", "0.50"))
USE_TTA = os.getenv("DETECT_TTA", "0").lower() in {"1", "true", "yes"}

# Day/Night profile
PROFILE = os.getenv("DETECT_PROFILE", "auto")              # "day", "night", or "auto"
DAY_CONF = float(os.getenv("DETECT_DAY_CONF", "0.25"))
NIGHT_CONF = float(os.getenv("DETECT_NIGHT_CONF", "0.15"))

# Bound the amount of work each run
MAX_FRAMES_PER_RUN = int(os.getenv("MAX_FRAMES_PER_RUN", "300"))

# JPEG sanity
MIN_JPEG_BYTES = int(os.getenv("MIN_JPEG_BYTES", "5000"))  # skip tiny/incomplete JPEGs

# [065] -------- Model --------
DETECTION_MODEL = YOLO(MODEL_NAME)

# [070] -------- SIGTERM handling --------
stop_requested = False
def _handle_sigterm(signum, frame):
    global stop_requested
    stop_requested = True
signal.signal(signal.SIGTERM, _handle_sigterm)

# [078] -------- Helpers: ROI --------
def load_roi_polygon() -> list[tuple[int, int]] | None:
    """
    Load polygon from meta/roi.json:
    { "polygon": [[x1,y1], [x2,y2], ...] }
    """
    if not ROI_JSON_PATH.exists():
        return None
    try:
        data = json.loads(ROI_JSON_PATH.read_text())
        pts = data.get("polygon", [])
        if not pts:
            return None
        return [(int(x), int(y)) for x, y in pts]
    except Exception:
        return None

def apply_roi_mask(img: np.ndarray, polygon: list[tuple[int, int]]) -> np.ndarray:
    """Return image masked to polygon (areas outside set to 0)."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return cv2.bitwise_and(img, img, mask=mask)

# [100] -------- Helpers: brightness / profile --------
def estimate_brightness_bgr(img: np.ndarray) -> float:
    """Cheap brightness estimate: mean of V channel in HSV (0..255)."""
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float(v.mean())

def choose_confidence_for_frame(base_conf: float, img: np.ndarray) -> float:
    """Pick confidence threshold based on PROFILE and image brightness."""
    if PROFILE == "day":
        return DAY_CONF
    if PROFILE == "night":
        return NIGHT_CONF
    # auto
    brightness = estimate_brightness_bgr(img)
    return DAY_CONF if brightness > 80.0 else NIGHT_CONF

# [115] -------- Helpers: CSV & scheduling --------
def load_already_processed_frames() -> set[str]:
    """
    Return a set of frame filenames that already appear in detections.csv.
    If CSV doesn't exist yet, return empty set.
    """
    if not DETECTIONS_CSV_PATH.exists():
        return set()
    processed: set[str] = set()
    with open(DETECTIONS_CSV_PATH, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            frame_name = row.get("frame_file")
            if frame_name:
                processed.add(frame_name)
    return processed

def newest_pending_frames(limit: int) -> list[Path]:
    """
    Collect *.jpg in samples/, drop ones already processed,
    and return only the newest `limit` frames.
    """
    all_jpgs = sorted(SAMPLES_DIR.glob("*.jpg"))
    if not all_jpgs:
        return []
    already_processed = load_already_processed_frames()
    pending = [p for p in all_jpgs if p.name not in already_processed]
    if not pending:
        return []
    pending.sort()
    return pending[-limit:]

# [140] -------- Drawing --------
def draw_label_on_frame(frame_image, label_text, left_x, top_y):
    """Draw a text label just above (x, y) on the frame."""
    cv2.putText(
        frame_image,
        label_text,
        (left_x, max(0, top_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        lineType=cv2.LINE_AA,
    )

# [152] -------- Main --------
def main():
    roi_polygon = load_roi_polygon()

    pending_frames = newest_pending_frames(MAX_FRAMES_PER_RUN)
    if not pending_frames:
        total_on_disk = len(list(SAMPLES_DIR.glob("*.jpg")))
        print(f"No new frames to process (samples on disk: {total_on_disk}).")
        return

    write_header = not DETECTIONS_CSV_PATH.exists()
    frames_with_cat = 0
    total_processed = 0

    # Open CSV in line-buffered mode; flush + fsync every row
    with open(DETECTIONS_CSV_PATH, "a", newline="", buffering=1) as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(
                ["frame_file", "det_idx", "label", "conf", "x1", "y1", "x2", "y2", "crop_file"]
            )
            csv_file.flush()
            try:
                os.fsync(csv_file.fileno())
            except OSError:
                pass

        for sample_path in tqdm(pending_frames, desc="Detecting", unit="frame"):
            if stop_requested:
                print("Stop requested; exiting after current frame.")
                break

            # Skip tiny/incomplete JPEGs
            try:
                if sample_path.stat().st_size < MIN_JPEG_BYTES:
                    continue
            except FileNotFoundError:
                # Was removed while we were processing
                continue

            frame_image = cv2.imread(str(sample_path))
            if frame_image is None or frame_image.size == 0:
                continue

            # ROI mask (if present)
            if roi_polygon:
                frame_image = apply_roi_mask(frame_image, roi_polygon)

            # Pick confidence for this frame (auto day/night)
            conf_used = choose_confidence_for_frame(CONFIDENCE_THRESHOLD_DEFAULT, frame_image)

            # Predict
            results = DETECTION_MODEL.predict(
                source=frame_image,
                device="cpu",
                verbose=False,
                conf=conf_used,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                augment=USE_TTA,
            )
            result = results[0]
            cats_in_this_frame = 0

            if result.boxes is not None:
                for det_index, det_box in enumerate(result.boxes):
                    class_id = int(det_box.cls)
                    class_label = DETECTION_MODEL.names[class_id]
                    if class_label != "cat":
                        continue

                    confidence = float(det_box.conf)
                    x_min, y_min, x_max, y_max = map(int, det_box.xyxy[0])

                    # Write crop
                    crop_img = frame_image[y_min:y_max, x_min:x_max]
                    if crop_img.size > 0:
                        crop_name = f"{sample_path.stem}_det{det_index}_{confidence:.2f}.jpg"
                        crop_path = CROPS_DIR / crop_name
                        cv2.imwrite(str(crop_path), crop_img)

                        # Record detection row
                        writer.writerow(
                            [
                                sample_path.name,
                                det_index,
                                class_label,
                                f"{confidence:.4f}",
                                x_min,
                                y_min,
                                x_max,
                                y_max,
                                crop_name,
                            ]
                        )
                        csv_file.flush()
                        try:
                            os.fsync(csv_file.fileno())
                        except OSError:
                            pass

                    # Annotate frame
                    cv2.rectangle(frame_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    draw_label_on_frame(frame_image, f"{class_label} {confidence:.2f}", x_min, y_min)
                    cats_in_this_frame += 1

            # Save annotated frame (even if zero cats, for debugging)
            out_frame_path = FRAMES_DIR / sample_path.name
            cv2.imwrite(str(out_frame_path), frame_image)

            total_processed += 1
            if cats_in_this_frame > 0:
                frames_with_cat += 1

    print(
        f"✅ Detection run complete. Processed {total_processed} new frames. "
        f"Frames with ≥1 cat: {frames_with_cat}."
    )
    print(f"- Annotated frames → {FRAMES_DIR}")
    print(f"- Crops → {CROPS_DIR}")
    print(f"- Detections CSV → {DETECTIONS_CSV_PATH}")
    if roi_polygon:
        print(f"- ROI active from {ROI_JSON_PATH}")

# [260] -------- Entrypoint --------
if __name__ == "__main__":
    main()