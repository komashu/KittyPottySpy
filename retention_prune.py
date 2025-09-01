#!/usr/bin/env python3
from pathlib import Path
import time

# --- Config ---
DIRECTORIES_TO_PRUNE = ["samples", "frames", "frames_labeled", "crops", "predictions"]
MAX_AGE_DAYS = 7
MAX_AGE_SECONDS = MAX_AGE_DAYS * 24 * 60 * 60
SOFT_CAP_BYTES = 1_500_000_000  # ~1.5 GB

def get_total_bytes() -> int:
    total_bytes = 0
    for directory in DIRECTORIES_TO_PRUNE:
        path_object = Path(directory)
        if not path_object.exists():
            continue
        for candidate_file in path_object.rglob("*"):
            if candidate_file.is_file():
                total_bytes += candidate_file.stat().st_size
    return total_bytes

def remove_old_files() -> int:
    """Delete files older than MAX_AGE_DAYS. Returns count removed."""
    files_removed = 0
    current_time = time.time()
    cutoff_epoch = current_time - MAX_AGE_SECONDS

    for directory in DIRECTORIES_TO_PRUNE:
        path_object = Path(directory)
        if not path_object.exists():
            continue
        for candidate_file in path_object.rglob("*"):
            if candidate_file.is_file() and candidate_file.stat().st_mtime < cutoff_epoch:
                try:
                    candidate_file.unlink()
                    files_removed += 1
                except Exception:
                    # intentionally quiet for cron; keep going
                    pass
    return files_removed

def enforce_soft_cap() -> int:
    """If total size exceeds SOFT_CAP_BYTES, delete oldest files first until under cap.
    Returns count removed."""
    files_removed = 0
    candidate_files = []

    for directory in DIRECTORIES_TO_PRUNE:
        path_object = Path(directory)
        if not path_object.exists():
            continue
        for candidate_file in path_object.rglob("*"):
            if candidate_file.is_file():
                candidate_files.append((candidate_file.stat().st_mtime, candidate_file))

    candidate_files.sort()  # oldest first
    total_bytes = get_total_bytes()
    index = 0

    while total_bytes > SOFT_CAP_BYTES and index < len(candidate_files):
        _, file_path = candidate_files[index]
        try:
            file_size = file_path.stat().st_size
            file_path.unlink()
            total_bytes -= file_size
            files_removed += 1
        except Exception:
            pass
        index += 1

    return files_removed

if __name__ == "__main__":
    removed_by_age = remove_old_files()
    removed_for_cap = enforce_soft_cap()
    print(
        f"Pruned files — by age: {removed_by_age}, by cap: {removed_for_cap}. "
        f"Current total: {get_total_bytes()} bytes."
    )
