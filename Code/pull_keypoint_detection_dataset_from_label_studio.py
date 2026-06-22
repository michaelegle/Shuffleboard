import shutil
import json
import os
import csv
from api_info import *
from label_studio_sdk import Client, LabelStudio
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

ls_client = Client(url = LABEL_STUDIO_API_URL, api_key = LABEL_STUDIO_API_KEY)

project = ls_client.get_project(id = LABEL_STUDIO_PROJECT_ID)

export_annotations = project.export_tasks(
    export_location = '../Data/keypoint_detection/dataset.json',
    download_all_tasks = False,
    export_type='JSON'
)

JSON_PATH = "../Data/keypoint_detection/dataset.json"
ORIGINAL_IMAGE_DIR = "../Training Images"
LABEL_OUTPUT = "../Data/keypoint_detection/labels"
IMAGE_OUTPUT = "../Data/keypoint_detection/images"



KEYPOINT_ORDER = [
    'Left 1 Point Line', 'Right 1 Point Line',
    'Left 2 Point Line', 'Right 2 Point Line',
    'Left 3 Point Line', 'Right 3 Point Line',
    'Left Baseline', 'Right Baseline',
]

os.makedirs(LABEL_OUTPUT, exist_ok=True)
os.makedirs(IMAGE_OUTPUT, exist_ok=True)


with open(JSON_PATH) as f:
    data = json.load(f)


total_num_items = len(data)

rows = []
skipped = 0
skipped_no_points = 0
print(total_num_items)


def copy_image(args):
    src, dst = args
    shutil.copy(src, dst)

image_files = [
    (os.path.join(ORIGINAL_IMAGE_DIR, f), os.path.join(IMAGE_OUTPUT, f))
    for f in os.listdir(ORIGINAL_IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

with ThreadPoolExecutor() as executor:
    executor.map(copy_image, image_files)

for task in data:
    
    image_filename = os.path.basename(task["data"]["image"])
    results = task.get("annotations", [{}])[0].get("result", [])
    keypoints = {}

    filename = os.path.basename(image_filename)
    # remove prefix
    if "IMG_" in filename:
        filename = filename.split("IMG_")[1]
        filename = "IMG_" + filename

    for item in results:
        if item["type"] != "keypointlabels":
            continue
        kp_name = item["value"]["keypointlabels"][0]
        # ignore the stones, just take the keypoints
        if kp_name not in KEYPOINT_ORDER:
            continue
        keypoints[kp_name] = (
            item["value"]["x"] / 100,
            item["value"]["y"] / 100,
        )
    
    # skip images where any expected keypoint is missing
    if len(keypoints) < len(KEYPOINT_ORDER):
        missing = [k for k in KEYPOINT_ORDER if k not in keypoints]
        print(f"Skipping {filename} — missing: {missing}")
        skipped += 1
        continue

    # build row: image path followed by x1,y1,x2,y2,...
    row = [os.path.join(IMAGE_OUTPUT, filename)]
    for kp_name in KEYPOINT_ORDER:
        x, y = keypoints[kp_name]
        row += [f"{x:.6f}", f"{y:.6f}"]
    rows.append(row)

if not rows:
    print("No valid annotations found.")

# split into train/val
train_rows, val_rows = train_test_split(rows, test_size = 0.2, random_state = 30)
header = ["image"]

for kp in KEYPOINT_ORDER:
    header += [f"{kp}_x", f"{kp}_y"]

for split, data in [("train", train_rows), ("val", val_rows)]:
    path = os.path.join(LABEL_OUTPUT, f"{split}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"{split}: {len(data)} images -> {path}")

print(f"Skipped {skipped} incomplete annotations")
print(f"Skipped {skipped_no_points} annotations with no labels at all")
