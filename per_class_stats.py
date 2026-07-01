import torch
import sys
from collections import defaultdict

sys.path.append("/content/drive/MyDrive/FasterRCNN_OBB_Project")

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

from validation_dataset import (
    ValidationDataset
)

from config import ID_TO_CLASS


# ------------------------
# SETTINGS
# ------------------------

NUM_CLASSES = 16

MODEL_PATH = (
    "/content/drive/MyDrive/FasterRCNN_OBB_Project/frcnn_epoch_95.pth"
)

CONFIDENCE_THRESHOLD = 0.30

# ------------------------

print("Loading Dataset...")

dataset = ValidationDataset()

NUM_IMAGES = len(dataset)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)

print(
    "Validation Images:",
    NUM_IMAGES
)

model = fasterrcnn_resnet50_fpn(
    weights=None
)

in_features = (
    model.roi_heads
         .box_predictor
         .cls_score
         .in_features
)

model.roi_heads.box_predictor = (
    FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )
)

print("Loading Model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model Loaded!")

class_counts = defaultdict(int)

for idx in range(NUM_IMAGES):

    if idx % 50 == 0:

        print(
            f"Processing {idx}/{NUM_IMAGES}"
        )

    image, target = dataset[idx]

    image = image.to(device)

    with torch.no_grad():

        predictions = model(
            [image]
        )

    labels = (
        predictions[0]["labels"]
        .cpu()
        .numpy()
    )

    scores = (
        predictions[0]["scores"]
        .cpu()
        .numpy()
    )

    for label, score in zip(
        labels,
        scores
    ):

        if score < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(label) - 1

        if class_id in ID_TO_CLASS:

            class_name = (
                ID_TO_CLASS[class_id]
            )

            class_counts[
                class_name
            ] += 1

print()
print(
    "===== PER CLASS DETECTIONS ====="
)
print()

for class_name, count in sorted(
    class_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{class_name:20s} : {count}"
    )

print()

with open(

    "/content/drive/MyDrive/FasterRCNN_OBB_Project/per_class_results_validation.txt",
    "w"
) as f:

    f.write(
        "===== PER CLASS DETECTIONS =====\n\n"
    )

    for class_name, count in sorted(
        class_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        f.write(
            f"{class_name:20s} : {count}\n"
        )

print(
    "Results Saved!"
)