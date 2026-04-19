import shutil
import json
import os
from api_info import *
from label_studio_sdk import Client, LabelStudio

ls_client = Client(url = LABEL_STUDIO_API_URL, api_key = LABEL_STUDIO_API_KEY)

project = ls_client.get_project(id = LABEL_STUDIO_PROJECT_ID)

export_annotations = project.export_tasks(
    export_location = '../Data/dataset.json',
    download_all_tasks = False,
    export_type='JSON'
)

JSON_PATH = "../Data/dataset.json"
ORIGINAL_IMAGE_DIR = "../Training Images"
TRAIN_LABEL_OUTPUT_DIR = "../Data/labels/train"
VAL_LABEL_OUTPUT_DIR = "../Data/labels/val"
TRAIN_IMAGE_OUTPUT_DIR = "../Data/images/train"
VAL_IMAGE_OUTPUT_DIR = "../Data/images/val"

CLASS_MAP = {
    "Black Stone": 0,
    "Gray Stone": 1,
    "Green Stone": 2
}

BOX_SIZE = 0.02  # tweak this (important!)

os.makedirs(TRAIN_LABEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAIN_IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_IMAGE_OUTPUT_DIR, exist_ok=True)


with open(JSON_PATH) as f:
    data = json.load(f)


# The items are labeled as a keypoint problem but this converts it to be a small bounding box to make it easier to train the model

total_num_items = len(data)

iter = 0
for item in data:
    iter = iter + 1
    image_path = item["data"]["image"]
    filename = os.path.basename(image_path)
    # remove prefix
    if "IMG_" in filename:
        filename = filename.split("IMG_")[1]
        filename = "IMG_" + filename
    txt_name = filename.replace(".jpg", ".txt")

    lines = []

    annotations = item.get("annotations", [])
    if not annotations:
        continue

    results = annotations[0]["result"]

    for r in results:
        if r["type"] != "keypointlabels":
            continue

        label = r["value"]["keypointlabels"][0]
        if label not in CLASS_MAP:
            continue

        x = r["value"]["x"] / 100.0
        y = r["value"]["y"] / 100.0

        cls = CLASS_MAP[label]

        lines.append(f"{cls} {x} {y} {BOX_SIZE} {BOX_SIZE}")

    if lines:
        if iter <= 0.8 * total_num_items:
            with open(os.path.join(TRAIN_LABEL_OUTPUT_DIR, txt_name), "w") as f:
                f.write("\n".join(lines))

            src_image_path = os.path.join(ORIGINAL_IMAGE_DIR, filename)
            dest_image_path = os.path.join(TRAIN_IMAGE_OUTPUT_DIR, filename)
            shutil.copy(src_image_path, dest_image_path)
        else:
            with open(os.path.join(VAL_LABEL_OUTPUT_DIR, txt_name), "w") as f:
                f.write("\n".join(lines))
            src_image_path = os.path.join(ORIGINAL_IMAGE_DIR, filename)
            dest_image_path = os.path.join(VAL_IMAGE_OUTPUT_DIR, filename)
            shutil.copy(src_image_path, dest_image_path)

        

print('done')



