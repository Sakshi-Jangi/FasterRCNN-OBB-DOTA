import matplotlib.pyplot as plt

epochs = []
losses = []

with open(
    "/content/drive/MyDrive//FasterRCNN_OBB_Project/loss_log.txt",
    "r"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        epoch, loss = line.split(",")

        epochs.append(
            int(epoch)
        )

        losses.append(
            float(loss)
        )

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    losses,
    marker="o"
)

plt.title(
    "Faster R-CNN Training Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.grid(True)

plt.savefig(
    "/content/drive/MyDrive//FasterRCNN_OBB_Project/loss_curve.png"
)

plt.show()

print(
    "Loss Curve Saved!"
)