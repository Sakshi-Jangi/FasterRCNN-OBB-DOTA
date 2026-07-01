import os
import cv2
import torch
import random

from torch.utils.data import Dataset

from config import CLASS_TO_ID


class FRCNNDataset(Dataset):

    def __init__(self):

        self.label_folder = (
            "/content/drive/MyDrive/FasterRCNN_OBB_Project/FasterRCNN_Dataset/train_labels"
        )

        self.image_folders = [
            "/content/drive/MyDrive/FasterRCNN_OBB_Project/FasterRCNN_Dataset/train_images",
            "/content/drive/MyDrive/FasterRCNN_OBB_Project/FasterRCNN_Dataset/train_images2/images"
        ]

        self.samples = []

        for image_folder in self.image_folders:

            print()
            print(
                "Checking Folder:",
                image_folder
            )

            image_files = sorted(
                os.listdir(image_folder)
            )

            print(
                "Images Found:",
                len(image_files)
            )

            count = 0

            for image_name in image_files:

                count += 1

                if count % 100 == 0:

                    print(
                        f"Processed {count} images"
                    )

                if not image_name.endswith(".png"):
                    continue

                label_path = os.path.join(
                    self.label_folder,
                    image_name.replace(
                        ".png",
                        ".txt"
                    )
                )

                if not os.path.exists(
                    label_path
                ):
                    continue

                with open(
                    label_path,
                    "r"
                ) as f:

                    lines = f.readlines()

                if len(lines) < 3:
                    continue

                self.samples.append(
                    (
                        image_folder,
                        image_name
                    )
                )

            print(
                "Valid Samples So Far:",
                len(self.samples)
            )

        print()
        print(
            "Total Training Images:",
            len(self.samples)
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        image_folder, image_name = self.samples[idx]

        image_path = os.path.join(
            image_folder,
            image_name
        )

        label_path = os.path.join(
            self.label_folder,
            image_name.replace(".png", ".txt")
        )

        image = cv2.imread(image_path)

        if image is None:

            print(
                "SKIPPING IMAGE:",
                image_path
            )

            return self.__getitem__(
                (idx + 1) % len(self.samples)
            )

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

        h, w = image.shape[:2]

        # Horizontal Flip
        if random.random() < 0.5:

            image = cv2.flip(
                image,
                1
            )

            new_boxes = []

            for box in boxes:

                xmin, ymin, xmax, ymax = box

                new_xmin = w - xmax
                new_xmax = w - xmin

                new_boxes.append(
                    [
                        new_xmin,
                        ymin,
                        new_xmax,
                        ymax
                    ]
                )

            boxes = new_boxes

        # Vertical Flip
        if random.random() < 0.5:

            image = cv2.flip(
                image,
                0
            )

            new_boxes = []

            for box in boxes:

                xmin, ymin, xmax, ymax = box

                new_ymin = h - ymax
                new_ymax = h - ymin

                new_boxes.append(
                    [
                        xmin,
                        new_ymin,
                        xmax,
                        new_ymax
                    ]
                )

            boxes = new_boxes

        # Brightness / Contrast
        if random.random() < 0.5:

            alpha = random.uniform(
                0.8,
                1.2
            )

            beta = random.randint(
                -20,
                20
            )

            image = cv2.convertScaleAbs(
                image,
                alpha=alpha,
                beta=beta
            )

        # Gaussian Blur
        if random.random() < 0.15:

            image = cv2.GaussianBlur(
                image,
                (3, 3),
                0
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

    dataset = FRCNNDataset()

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

    if len(target["boxes"]) > 0:

        print(
            "First Box:",
            target["boxes"][0]
        )