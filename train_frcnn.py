import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from torch.utils.data import DataLoader

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

from detection_dataset_frcnn import (
    FRCNNDataset
)


# ---------------------------------
# SETTINGS
# ---------------------------------

NUM_CLASSES = 16

START_EPOCH = 86
END_EPOCH = 95

MODEL_PATH = (
    "/content/drive/MyDrive/FasterRCNN_OBB_Project/frcnn_epoch_85.pth"
)

LEARNING_RATE = 0.00005

# ---------------------------------


def collate_fn(batch):

    return tuple(
        zip(*batch)
    )


dataset = FRCNNDataset()

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

print(
    "Total Training Images:",
    len(dataset)
)


model = fasterrcnn_resnet50_fpn(
    weights=None,
    min_size=700,
    max_size=1200
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
# Allow more detections per image
model.roi_heads.detections_per_img = 300

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

print(
    "Epoch 85 Model Loaded!"
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)

model.to(device)

torch.cuda.empty_cache()

model.train()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)

for epoch in range(
    START_EPOCH,
    END_EPOCH + 1
):

    total_loss = 0

    print()
    print(
        f"========== EPOCH {epoch}/{END_EPOCH} =========="
    )

    print(
        "Current LR:",
        optimizer.param_groups[0]["lr"]
    )

    for batch_idx, (images, targets) in enumerate(loader):

        images = [
            img.to(device)
            for img in images
        ]

        targets = [
            {
                k: v.to(device)
                for k, v in t.items()
            }
            for t in targets
        ]

        loss_dict = model(
            images,
            targets
        )

        losses = sum(
            loss
            for loss in loss_dict.values()
        )

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        total_loss += losses.item()

        if batch_idx % 20 == 0:

            print(
                f"Batch {batch_idx}/{len(loader)} | Loss: {losses.item():.4f}"
            )

    avg_loss = (
        total_loss /
        len(loader)
    )

    print()

    print(
        f"Epoch {epoch}"
    )

    print(
        f"Average Loss: {avg_loss:.4f}"
    )

    with open(
        "/content/drive/MyDrive/FasterRCNN_OBB_Project/loss_log.txt",
        "a"
    ) as f:

        f.write(
            f"{epoch},{avg_loss}\n"
        )

    torch.save(
        model.state_dict(),
        f"/content/drive/MyDrive/FasterRCNN_OBB_Project/frcnn_epoch_{epoch}.pth"
    )

    print(
        f"Checkpoint Saved: Epoch {epoch}"
    )

    scheduler.step()

    torch.cuda.empty_cache()

torch.save(
    model.state_dict(),
    "/content/drive/MyDrive/FasterRCNN_OBB_Project/frcnn_dota_full.pth"
)

print()

print(
    "Training Complete!"
)

print(
    "Final Model Saved!"
)