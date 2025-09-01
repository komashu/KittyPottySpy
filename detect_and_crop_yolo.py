import csv
from pathlib import Path
import cv2
from ultralytics import YOLO
import os, warnings
import torch

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_num_threads(3)


SAMPLES = Path("samples")
FRAMES  = Path("frames")
CROPS   = Path("crops")
META    = Path("meta")
META.mkdir(exist_ok=True)
FRAMES.mkdir(exist_ok=True)
CROPS.mkdir(exist_ok=True)

# model: "s" (small) is more accurate than "n" (nano)
model = YOLO("yolov8s.pt")

# params to increase recall
CONF_THRES = 0.20
IMG_SIZE   = 640
IOU_THRES  = 0.5
AUGMENT    = True

csv_path = META / "detections.csv"

def draw_label(img, text, x, y):
    cv2.putText(img, text, (x, max(0, y-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, lineType=cv2.LINE_AA)

jpgs = sorted(list(SAMPLES.glob("*.jpg")))
hits = 0
total = 0

# write header if file doesn't exist
write_header = not csv_path.exists()
with open(csv_path, "a", newline="") as fcsv:
    writer = csv.writer(fcsv)
    if write_header:
        writer.writerow(["frame_file","det_idx","label","conf","x1","y1","x2","y2","crop_file"])

    for img_path in jpgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(
            source=img,
            device="cpu",
            verbose=False,
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=IMG_SIZE,
            augment=AUGMENT
        )
        det = results[0]
        n_cats = 0

        if det.boxes is not None:
            for i, box in enumerate(det.boxes):
                cls_id = int(box.cls)
                label = model.names[cls_id]
                conf  = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == "cat":
                    # crop
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_file = CROPS / f"{img_path.stem}_det{i}_{conf:.2f}.jpg"
                        cv2.imwrite(str(crop_file), crop)
                        # record metadata row
                        writer.writerow([
                            img_path.name, i, label, f"{conf:.4f}", x1, y1, x2, y2, crop_file.name
                        ])
                    # annotate detection on the frame
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    draw_label(img, f"{label} {conf:.2f}", x1, y1)
                    n_cats += 1

        out_file = FRAMES / img_path.name
        cv2.imwrite(str(out_file), img)

        total += 1
        if n_cats > 0:
            hits += 1

print(f"✅ Detection done. Processed {total} frames. Frames with ≥1 cat: {hits}.")
print(f"- Annotated frames → {FRAMES}")
print(f"- Crops → {CROPS}")
print(f"- Detections CSV → {csv_path}")
