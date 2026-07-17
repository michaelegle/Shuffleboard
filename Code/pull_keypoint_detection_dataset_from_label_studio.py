import shutil
import json
import os
import random
from pathlib import Path
from collections import Counter
 
from api_info import *
from label_studio_sdk import Client, LabelStudio
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
 
JSON_PATH = Path("Data/keypoint_detection/dataset.json")
ORIGINAL_IMAGE_DIR = Path("Training Images/")
 
OUTPUT_ROOT = Path("Data/keypoint_detection")
TRAIN_LABEL_DIR = OUTPUT_ROOT / "labels/train"
VAL_LABEL_DIR = OUTPUT_ROOT / "labels/val"
TRAIN_IMAGE_DIR = OUTPUT_ROOT / "images/train"
VAL_IMAGE_DIR = OUTPUT_ROOT / "images/val"
 
CLASS_MAP = {
    "Left 1 Point Line": 0,
    "Right 1 Point Line": 1,
    "Left 2 Point Line": 2,
    "Right 2 Point Line": 3,
    "Left 3 Point Line": 4,
    "Right 3 Point Line": 5,
    "Left Baseline": 6,
    "Right Baseline": 7,
}
 
BOX_SIZE = 0.02  # TODO tweak this at some point
VAL_FRACTION = 0.2
RANDOM_SEED = 30
 
# ---------------------------------------------------------------------------
# Step 1: pull the latest export from Label Studio
# ---------------------------------------------------------------------------
 
def export_from_label_studio():
    ls_client = Client(url=LABEL_STUDIO_API_URL, api_key=LABEL_STUDIO_API_KEY)
    project = ls_client.get_project(id=LABEL_STUDIO_PROJECT_ID)
 
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
 
    project.export_tasks(
        export_location=str(JSON_PATH),
        download_all_tasks=True,  # True = every task in the project, not just
                                   # whatever's in the currently active tab/view
        export_type="JSON",
    )
 
 
# ---------------------------------------------------------------------------
# Step 2: normalize a Label Studio image path to the local filename
# ---------------------------------------------------------------------------
 
def resolve_filename(image_path: str) -> str:
    filename = os.path.basename(image_path)
    if "IMG_" in filename:
        filename = "IMG_" + filename.split("IMG_")[1]
    return filename
 
 
# ---------------------------------------------------------------------------
# Step 3: convert one task's annotations into YOLO label lines
# ---------------------------------------------------------------------------
 
def task_to_yolo_lines(item, stats: Counter):
    """Returns a list of YOLO label lines for a task, pulling keypoints from
    ALL annotations on the task (not just the first one)."""
    annotations = item.get("annotations", [])
    if not annotations:
        stats["no_annotation"] += 1
        return []
 
    lines = []
    for ann in annotations:
        if ann.get("was_cancelled"):
            continue
        for r in ann.get("result", []):
            if r.get("type") != "keypointlabels":
                continue
 
            label = r["value"]["keypointlabels"][0]
            if label not in CLASS_MAP:
                stats["unknown_label"] += 1
                continue
 
            x = r["value"]["x"] / 100.0
            y = r["value"]["y"] / 100.0
            cls = CLASS_MAP[label]
            lines.append(f"{cls} {x} {y} {BOX_SIZE} {BOX_SIZE}")
 
    if not lines:
        stats["empty_after_parse"] += 1
 
    return lines
 
 
# ---------------------------------------------------------------------------
# Step 4: build the dataset
# ---------------------------------------------------------------------------
 
def reset_output_dirs():
    for d in (TRAIN_LABEL_DIR, VAL_LABEL_DIR, TRAIN_IMAGE_DIR, VAL_IMAGE_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
 
 
def build_dataset():
    reset_output_dirs()
 
    with open(JSON_PATH) as f:
        data = json.load(f)
 
    stats = Counter()
    stats["total_tasks"] = len(data)
 
    # First pass: figure out which tasks are usable at all, so the
    # train/val split is computed over the *real* eligible set.
    usable_items = []  # (filename, lines)
    seen_filenames = {}
 
    for item in data:
        filename = resolve_filename(item["data"]["image"])
 
        lines = task_to_yolo_lines(item, stats)
        if not lines:
            continue
 
        src_image_path = ORIGINAL_IMAGE_DIR / filename
        if not src_image_path.exists():
            stats["missing_source_image"] += 1
            print(f"  [!] missing source image, skipping: {src_image_path}")
            continue
 
        if filename in seen_filenames:
            stats["duplicate_filename"] += 1
            print(f"  [!] duplicate filename, keeping first: {filename}")
            continue
        seen_filenames[filename] = True
 
        usable_items.append((filename, lines))
 
    # Randomized, seeded split
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(usable_items)
 
    n_val = int(len(usable_items) * VAL_FRACTION)
    val_items = usable_items[:n_val]
    train_items = usable_items[n_val:]
 
    for split_name, items, label_dir, image_dir in (
        ("train", train_items, TRAIN_LABEL_DIR, TRAIN_IMAGE_DIR),
        ("val", val_items, VAL_LABEL_DIR, VAL_IMAGE_DIR),
    ):
        for filename, lines in items:
            txt_name = filename.replace(".jpg", ".txt")
            (label_dir / txt_name).write_text("\n".join(lines))
            shutil.copy(ORIGINAL_IMAGE_DIR / filename, image_dir / filename)
        stats[f"written_{split_name}"] = len(items)
 
    # ---- summary ----
    print("\n--- Dataset build summary ---")
    print(f"Total tasks in export:       {stats['total_tasks']}")
    print(f"No annotation at all:        {stats['no_annotation']}")
    print(f"Empty after parsing labels:  {stats['empty_after_parse']}")
    print(f"Unrecognized label values:   {stats['unknown_label']}")
    print(f"Missing source image file:   {stats['missing_source_image']}")
    print(f"Duplicate filenames:         {stats['duplicate_filename']}")
    print(f"Written to train:            {stats['written_train']}")
    print(f"Written to val:              {stats['written_val']}")
    print(f"Total written:               {stats['written_train'] + stats['written_val']}")
 
 
if __name__ == "__main__":
    print("Exporting latest annotations from Label Studio...")
    export_from_label_studio()
 
    print("Building train/val dataset...")
    build_dataset()
 
    print("\nDone.")