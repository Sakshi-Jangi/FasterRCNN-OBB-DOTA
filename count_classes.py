import os
from collections import Counter

from config import CLASS_TO_ID

LABEL_FOLDER = "/content/drive/MyDrive/FasterRCNN_Dataset/train_labels"

class_counts = Counter()

files = os.listdir(LABEL_FOLDER)

for file in files:

    if not file.endswith(".txt"):
        continue

    with open(
        os.path.join(LABEL_FOLDER, file),
        "r"
    ) as f:

        lines = f.readlines()[2:]

    for line in lines:

        parts = line.split()

        if len(parts) < 9:
            continue

        class_name = parts[8]

        if class_name in CLASS_TO_ID:

            class_counts[class_name] += 1

print("\n===== CLASS COUNTS =====\n")

total = 0

for cls, count in sorted(
    class_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{cls:20s} : {count}"
    )

    total += count

print("\nTotal Objects:", total)