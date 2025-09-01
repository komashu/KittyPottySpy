import csv
import os
import warnings
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# --- Runtime/env tuning (quiet + sane CPU threads) ---
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_num_threads(3)

# --- Paths ---
SAMPLES_DIR = Path("samples")
FRAMES_DIR = Path("frames")
CROPS_DIR = Path("crops")
META_DIR = Path("meta")

META_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
CROPS_DIR.mkdir(exist_ok=True)

DETECTIONS_CSV_PATH = META_DIR / "detections.csv"

# --- Model & inference params ---
# Use "s" for a bit more accuracy than "n" (you can swap to yolov8n.pt for speed)
DETECTION_MODEL = YOLO("yolov8s.pt")
CONFIDENCE_THRESHOLD = 0.20
IMAGE_SIZE = 640
IOU_THRESHOLD = 0.50
USE_TEST_TIME_AUGMENTATION = True


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


def main():
    sample_image_paths = sorted(SAMPLES_DIR.glob("*.jpg"))
    frames_with_at_least_one_cat = 0
    total_frames_processed = 0

    # open CSV once; write header if new
    write_header = not DETECTIONS_CSV_PATH.exists()
    with open(DETECTIONS_CSV_PATH, "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(
                [
                    "frame_file",
                    "det_idx",
                    "label",
                    "conf",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "crop_file",
                ]
            )

        for sample_path in sample_image_paths:
            frame_image = cv2.imread(str(sample_path))
            if frame_image is None:
                continue

            prediction_results = DETECTION_MODEL.predict(
                source=frame_image,
                device="cpu",
                verbose=False,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                augment=USE_TEST_TIME_AUGMENTATION,
            )
            detection_result = prediction_results[0]
            cats_found_in_this_frame = 0

            if detection_result.boxes is not None:
                for detection_index, detection_box in enumerate(detection_result.boxes):
                    class_id = int(detection_box.cls)
                    class_label = DETECTION_MODEL.names[class_id]
                    confidence_score = float(detection_box.conf)
                    x_min, y_min, x_max, y_max = map(int, detection_box.xyxy[0])

                    if class_label == "cat":
                        # write crop if valid
                        cat_crop = frame_image[y_min:y_max, x_min:x_max]
                        if cat_crop.size > 0:
                            crop_filename = (
                                CROPS_DIR
                                / f"{sample_path.stem}_det{detection_index}_{confidence_score:.2f}.jpg"
                            )
                            cv2.imwrite(str(crop_filename), cat_crop)

                            # record metadata
                            csv_writer.writerow(
                                [
                                    sample_path.name,
                                    detection_index,
                                    class_label,
                                    f"{confidence_score:.4f}",
                                    x_min,
                                    y_min,
                                    x_max,
                                    y_max,
                                    crop_filename.name,
                                ]
                            )

                        # annotate frame
                        cv2.rectangle(
                            frame_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
                        )
                        draw_label_on_frame(
                            frame_image, f"{class_label} {confidence_score:.2f}", x_min, y_min
                        )
                        cats_found_in_this_frame += 1

            output_frame_path = FRAMES_DIR / sample_path.name
            cv2.imwrite(str(output_frame_path), frame_image)

            total_frames_processed += 1
            if cats_found_in_this_frame > 0:
                frames_with_at_least_one_cat += 1

    print(
        f"✅ Detection done. Processed {total_frames_processed} frames. "
        f"Frames with ≥1 cat: {frames_with_at_least_one_cat}."
    )
    print(f"- Annotated frames → {FRAMES_DIR}")
    print(f"- Crops → {CROPS_DIR}")
    print(f"- Detections CSV → {DETECTIONS_CSV_PATH}")


if __name__ == "__main__":
    main()