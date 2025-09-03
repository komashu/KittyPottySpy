#!/usr/bin/env python3
import csv
from pathlib import Path

META = Path("meta")
SRC = META / "detections.csv"
DST = META / "detections_clean.csv"
BACKUP = META / "detections.csv.bak"

EXPECTED = ["frame_file","det_idx","label","conf","x1","y1","x2","y2","crop_file"]

def is_int(s):
    try: int(s); return True
    except: return False

def is_float(s):
    try: float(s); return True
    except: return False

if not SRC.exists():
    print(f"{SRC} not found.")
    raise SystemExit(1)

kept = 0
skipped = 0
with open(SRC, newline="") as fin, open(DST, "w", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=EXPECTED)
    writer.writeheader()
    for row in reader:
        try:
            # basic sanity checks
            if row.get("label") != "cat":
                continue
            if not (is_int(row["det_idx"]) and is_float(row["conf"]) and
                    is_int(row["x1"]) and is_int(row["y1"]) and is_int(row["x2"]) and is_int(row["y2"])):
                skipped += 1
                continue
            writer.writerow({
                "frame_file": row["frame_file"],
                "det_idx": int(row["det_idx"]),
                "label": row["label"],
                "conf": float(row["conf"]),
                "x1": int(row["x1"]), "y1": int(row["y1"]), "x2": int(row["x2"]), "y2": int(row["y2"]),
                "crop_file": row["crop_file"]
            })
            kept += 1
        except Exception:
            skipped += 1

print(f"Kept {kept} rows, skipped {skipped} malformed rows.")
# swap files
if kept > 0:
    SRC.rename(BACKUP)
    DST.rename(SRC)
    print(f"Wrote clean CSV → {SRC} (backup at {BACKUP})")
else:
    print("No valid rows found; leaving original file untouched.")