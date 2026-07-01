import torch
import sys
import importlib.util

sys.path.append("/content/drive/MyDrive/FasterRCNN_OBB_Project")

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

spec = importlib.util.spec_from_file_location(
    "validation_dataset",
    "/content/drive/MyDrive/FasterRCNN_OBB_Project/validation_dataset.py"
)

module = importlib.util.module_from_spec(spec)

spec.loader.exec_module(module)

ValidationDataset = module.ValidationDataset

from evaluate_image import (
    evaluate_image
)

# ----------------------------
# SETTINGS
# ----------------------------

NUM_CLASSES = 16

MODEL_PATH = (
    "/content/drive/MyDrive/FasterRCNN_OBB_Project/frcnn_epoch_95.pth"
)

CONFIDENCE_THRESHOLD = 0.30

FAST_MODE = False

# ----------------------------

print("Loading Dataset...")

dataset = ValidationDataset()

print(
    "Dataset Size:",
    len(dataset)
)

if FAST_MODE:

    NUM_IMAGES = min(
        200,
        len(dataset)
    )

else:

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
    "Images Used:",
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

print(
    "Loading Model..."
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

print(
    "Model Loaded!"
)

total_tp = 0
total_fp = 0
total_fn = 0

print(
    "Starting Evaluation..."
)

for idx in range(NUM_IMAGES):

    if idx % 50 == 0:

        print(
            f"Evaluating Image {idx}/{NUM_IMAGES}"
        )

    image, target = dataset[idx]

    image = image.to(device)

    with torch.no_grad():

        predictions = model(
            [image]
        )

    scores = (
        predictions[0]["scores"]
        .cpu()
        .numpy()
    )

    boxes = (
        predictions[0]["boxes"]
        .cpu()
        .numpy()
    )

    pred_boxes = []

    for box, score in zip(
        boxes,
        scores
    ):

        if score >= CONFIDENCE_THRESHOLD:

            pred_boxes.append(
                box
            )

    gt_boxes = (
        target["boxes"]
        .cpu()
        .numpy()
    )

    metrics = evaluate_image(
        pred_boxes,
        gt_boxes
    )

    total_tp += metrics["tp"]
    total_fp += metrics["fp"]
    total_fn += metrics["fn"]

precision = (
    total_tp /
    (total_tp + total_fp)
    if (total_tp + total_fp) > 0
    else 0
)

recall = (
    total_tp /
    (total_tp + total_fn)
    if (total_tp + total_fn) > 0
    else 0
)

f1_score = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)

accuracy = (
    total_tp /
    (total_tp + total_fp + total_fn)
    if (total_tp + total_fp + total_fn) > 0
    else 0
)

print()
print("===== RESULTS =====")
print()

print("TP:", total_tp)
print("FP:", total_fp)
print("FN:", total_fn)

print(
    "Precision:",
    precision
)

print(
    "Recall:",
    recall
)

print(
    "F1 Score:",
    f1_score
)

print(
    "Accuracy:",
    accuracy
)

with open(
    "/content/drive/MyDrive/FasterRCNN_OBB_Project/evaluation_results.txt",
    "a"
) as f:

    f.write(
        "\n=====================\n"
    )

    f.write(
        f"MODEL: {MODEL_PATH}\n"
    )

    f.write(
        f"CONFIDENCE: {CONFIDENCE_THRESHOLD}\n"
    )

    f.write(
        f"IMAGES: {NUM_IMAGES}\n"
    )

    f.write(
        f"TP: {total_tp}\n"
    )

    f.write(
        f"FP: {total_fp}\n"
    )

    f.write(
        f"FN: {total_fn}\n"
    )

    f.write(
        f"Precision: {precision}\n"
    )

    f.write(
        f"Recall: {recall}\n"
    )

    f.write(
        f"F1 Score: {f1_score}\n"
    )

    f.write(
        f"Accuracy: {accuracy}\n"
    )

print()
print(
    "Results Saved!"
)