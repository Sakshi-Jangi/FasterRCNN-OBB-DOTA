import os
import cv2
import torch

from torch.utils.data import Dataset
from config import CLASS_TO_ID


class ValidationDataset(Dataset):

    def __init__(self):

        self.label_folder = (
            "/content/drive/MyDrive/FasterRCNN_OBB_Project/FasterRCNN_Dataset/labels"
        )

        self.image_folder = (
            "/content/drive/MyDrive/FasterRCNN_OBB_Project/FasterRCNN_Dataset/images"
        )

        self.samples = []

        image_files = sorted(
            os.listdir(self.image_folder)
        )

        for image_name in image_files:

            if not image_name.endswith(".png"):
                continue

            label_path = os.path.join(
                self.label_folder,
                image_name.replace(
                    ".png",
                    ".txt"
                )
            )

            if not os.path.exists(label_path):
                continue

            with open(label_path, "r") as f:
                lines = f.readlines()

            if len(lines) < 3:
                continue

            self.samples.append(image_name)

        print(
            "Total Validation Images:",
            len(self.samples)
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        image_name = self.samples[idx]

        image_path = os.path.join(
            self.image_folder,
            image_name
        )

        label_path = os.path.join(
            self.label_folder,
            image_name.replace(
                ".png",
                ".txt"
            )
        )

        image = cv2.imread(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        boxes = []
        labels = []

        with open(label_path, "r") as f:

            lines = f.readlines()[2:]

        for line in lines:

            parts = line.split()

            if len(parts) < 9:
                continue

            coords = list(
                map(float, parts[:8])
            )

            xs = coords[0::2]
            ys = coords[1::2]

            xmin = min(xs)
            ymin = min(ys)

            xmax = max(xs)
            ymax = max(ys)

            class_name = parts[8]

            if class_name not in CLASS_TO_ID:
                continue

            boxes.append(
                [
                    xmin,
                    ymin,
                    xmax,
                    ymax
                ]
            )

            labels.append(
                CLASS_TO_ID[class_name] + 1
            )

        image = torch.tensor(
            image,
            dtype=torch.float32
        ) / 255.0

        image = image.permute(
            2,
            0,
            1
        )

        target = {
            "boxes": torch.tensor(
                boxes,
                dtype=torch.float32
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.int64
            )
        }

        return image, target


if __name__ == "__main__":

    dataset = ValidationDataset()

    print(
        "Dataset Size:",
        len(dataset)
    )

    image, target = dataset[0]

    print(
        "Image Shape:",
        image.shape
    )

    print(
        "Boxes Shape:",
        target["boxes"].shape
    )

    print(
        "Labels Shape:",
        target["labels"].shape
    )