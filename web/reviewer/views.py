from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import csv, shutil, time
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from .forms import ReviewActionForm
from .models import Feedback

PROJECT_ROOT = Path(settings.BASE_DIR)
CROPS_DIRECTORY = PROJECT_ROOT / "crops"
LABELED_FRAMES_DIRECTORY = PROJECT_ROOT / "frames_labeled"
FRAMES_DIRECTORY = PROJECT_ROOT / "frames"
PREDICTIONS_CSV_PATH = PROJECT_ROOT / "predictions" / "predictions.csv"
DETECTIONS_CSV_PATH = PROJECT_ROOT / "meta" / "detections.csv"
CATS_DIRECTORY = PROJECT_ROOT / "cats"

COPY_APPROVED_TO_GALLERY = True

def _load_crop_to_frame_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not DETECTIONS_CSV_PATH.exists():
        return mapping
    with open(DETECTIONS_CSV_PATH, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            mapping[row["crop_file"]] = row["frame_file"]
    return mapping

def _load_recent_predictions(limit_count: int = 60, offset_count: int = 0) -> List[Dict[str, Any]]:
    if not PREDICTIONS_CSV_PATH.exists():
        return []
    with open(PREDICTIONS_CSV_PATH, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    def crop_mtime(row: Dict[str, str]) -> float:
        crop_path = Path(row["crop_path"])
        return crop_path.stat().st_mtime if crop_path.exists() else 0.0
    sorted_rows = sorted(rows, key=crop_mtime, reverse=True)
    selected = sorted_rows[offset_count: offset_count + limit_count]
    items: List[Dict[str, Any]] = []
    crop_to_frame = _load_crop_to_frame_map()
    for row in selected:
        crop_path = Path(row["crop_path"])
        items.append({
            "crop_filename": crop_path.name,
            "predicted_label": row.get("predicted", "unknown"),
            "distance_value": row.get("cosine_distance", ""),
            "frame_filename": crop_to_frame.get(crop_path.name),
        })
    return items

def review_list_view(request: HttpRequest) -> HttpResponse:
    page_number = int(request.GET.get("page", "1"))
    page_size = int(request.GET.get("page_size", "30"))
    offset_count = max(page_number - 1, 0) * page_size
    items = _load_recent_predictions(limit_count=page_size, offset_count=offset_count)
    has_more = len(items) == page_size
    return render(request, "reviewer/review.html", {
        "items": items,
        "page_number": page_number,
        "page_size": page_size,
        "has_more": has_more,
    })

def review_action_view(request: HttpRequest) -> HttpResponse:
    if request.method != "POST":
        return redirect("review_list")
    form = ReviewActionForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid form submission.")
        return redirect("review_list")

    crop_filename = form.cleaned_data["crop_filename"]
    predicted_label = form.cleaned_data.get("predicted") or None
    distance_value = form.cleaned_data.get("distance_value")
    action_choice = form.cleaned_data["action"]
    override_label = (form.cleaned_data.get("override_label") or "").strip()

    corrected_label: Optional[str] = None
    decision: str = action_choice
    if action_choice == "approve":
        corrected_label = predicted_label
    elif action_choice == "unknown":
        corrected_label = None
    elif action_choice == "override":
        corrected_label = override_label if override_label else None
        if not corrected_label:
            decision = "unknown"

    Feedback.objects.create(
        created_at_epoch=int(time.time()),
        crop_filename=crop_filename,
        predicted_label=predicted_label,
        distance_value=float(distance_value) if distance_value not in (None, "") else None,
        decision=decision,
        corrected_label=corrected_label,
    )

    if decision in ("approve", "override") and corrected_label:
        source_path = CROPS_DIRECTORY / crop_filename
        destination_directory = CATS_DIRECTORY / corrected_label / "from_review"
        destination_directory.mkdir(parents=True, exist_ok=True)
        if source_path.exists():
            try:
                shutil.copy2(source_path, destination_directory / crop_filename)
            except Exception:
                pass

    messages.success(request, f"Saved: {crop_filename} → {decision}" + (f" ({corrected_label})" if corrected_label else ""))
    return redirect("review_list")

def labeled_frames_view(request: HttpRequest) -> HttpResponse:
    frame_names = sorted([p.name for p in LABELED_FRAMES_DIRECTORY.glob("*.jpg")], reverse=True)[:200]
    return render(request, "reviewer/labeled_frames.html", {"frame_names": frame_names})
