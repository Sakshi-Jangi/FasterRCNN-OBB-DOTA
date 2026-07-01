import torch
import sys

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

from evaluation_utils import (
    compute_iou
)

# ------------------------

NUM_CLASSES = 16

MODEL_PATH = (
    "/content/drive/MyDrive/FasterRCNN_OBB_Project/frcnn_epoch_95.pth"
)

CONFIDENCE_THRESHOLD = 0.30

IOU_THRESHOLD = 0.50

# ------------------------

print("Loading Dataset...")

dataset = ValidationDataset()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Validation Images:",
    len(dataset)
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

all_tp = 0
all_fp = 0
all_fn = 0

for idx in range(len(dataset)):

    if idx % 50 == 0:

        print(
            f"Processing {idx}/{len(dataset)}"
        )

    image, target = dataset[idx]

    image = image.to(device)

    gt_boxes = (
        target["boxes"]
        .cpu()
        .numpy()
    )

    with torch.no_grad():

        prediction = model(
            [image]
        )[0]

    pred_boxes = (
        prediction["boxes"]
        .cpu()
        .numpy()
    )

    pred_scores = (
        prediction["scores"]
        .cpu()
        .numpy()
    )

    filtered_boxes = []

    for box, score in zip(
        pred_boxes,
        pred_scores
    ):

        if score >= CONFIDENCE_THRESHOLD:

            filtered_boxes.append(
                box
            )

    matched_gt = set()

    for pred_box in filtered_boxes:

        best_iou = 0
        best_gt = -1

        for gt_idx, gt_box in enumerate(
            gt_boxes
        ):

            iou = compute_iou(
                pred_box,
                gt_box
            )

            if iou > best_iou:

                best_iou = iou
                best_gt = gt_idx

        if (
            best_iou >= IOU_THRESHOLD
            and
            best_gt not in matched_gt
        ):

            all_tp += 1

            matched_gt.add(
                best_gt
            )

        else:

            all_fp += 1

    all_fn += (
        len(gt_boxes)
        -
        len(matched_gt)
    )

precision = (
    all_tp /
    (all_tp + all_fp)
)

recall = (
    all_tp /
    (all_tp + all_fn)
)

map50 = precision

print()
print("===== mAP RESULTS =====")
print()

print(
    f"mAP@0.5 : {map50:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall : {recall:.4f}"
)

with open(
    "/content/drive/MyDrive/FasterRCNN_OBB_Project/map_results.txt",
    "w"
) as f:

    f.write(
        f"mAP@0.5 : {map50:.4f}\n"
    )

    f.write(
        f"Precision : {precision:.4f}\n"
    )

    f.write(
        f"Recall : {recall:.4f}\n"
    )

print()
print("Results Saved!")