import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = r"C:\Users\Lenovo\Downloads\TrainData"
PROJECT_ROOT = r"C:\Users\Lenovo\Desktop\Landslide_ai_sih"

MODEL_FILE = os.path.join(PROJECT_ROOT, "best_unet.pth")
STATS_FILE = os.path.join(PROJECT_ROOT, "normalization_stats.npz")

IMG_DIR = os.path.join(DATA_ROOT, "img")
MASK_DIR = os.path.join(DATA_ROOT, "mask")


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
THRESHOLD = 0.50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("LANDSLIDE SIZE-BASED VALIDATION ANALYSIS")
print("=" * 70)
print("Device:", DEVICE)


# ============================================================
# MODEL
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=14, out_channels=1):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(
            1024, 512, 2, stride=2
        )
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(
            512, 256, 2, stride=2
        )
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
        self.dec1 = DoubleConv(128, 64)

        self.output = nn.Conv2d(
            64, out_channels, 1
        )

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = UNet(
    in_channels=14,
    out_channels=1
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD NORMALIZATION
# ============================================================

stats = np.load(STATS_FILE)

means = stats["mean"]
stds = stats["std"]

print("Normalization statistics loaded.")


# ============================================================
# DATASET
# ============================================================

image_files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.endswith(".h5")
])

mask_files = sorted([
    f for f in os.listdir(MASK_DIR)
    if f.endswith(".h5")
])

assert len(image_files) == len(mask_files)

indices = list(range(len(image_files)))

random.seed(SEED)
random.shuffle(indices)

# EXACT same split used during training
split = 3040

train_indices = indices[:split]
val_indices = indices[split:]

print("\nDataset:")
print("Total:", len(indices))
print("Training:", len(train_indices))
print("Validation:", len(val_indices))


# ============================================================
# SIZE CATEGORIES
# ============================================================

categories = {
    "Tiny (<50)": [],
    "Small (50-200)": [],
    "Medium (200-500)": [],
    "Large (>500)": []
}


# ============================================================
# METRIC STORAGE
# ============================================================

results = {
    name: {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "missed": 0
    }
    for name in categories
}


# ============================================================
# PROCESS VALIDATION SET
# ============================================================

print("\nEvaluating validation set...")

for counter, index in enumerate(val_indices, start=1):

    img_path = os.path.join(
        IMG_DIR,
        image_files[index]
    )

    mask_path = os.path.join(
        MASK_DIR,
        mask_files[index]
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    with h5py.File(img_path, "r") as f:
        image = f["img"][:]

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    with h5py.File(mask_path, "r") as f:
        mask = f["mask"][:]

    mask = (mask > 0).astype(np.uint8)

    gt_pixels = int(mask.sum())

    # --------------------------------------------------------
    # Ignore patches with no landslide
    # --------------------------------------------------------

    if gt_pixels == 0:
        continue

    # --------------------------------------------------------
    # Determine size category
    # --------------------------------------------------------

    if gt_pixels < 50:
        category = "Tiny (<50)"

    elif gt_pixels <= 200:
        category = "Small (50-200)"

    elif gt_pixels <= 500:
        category = "Medium (200-500)"

    else:
        category = "Large (>500)"

    categories[category].append(index)

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image = (
        image - means
    ) / (
        stds + 1e-8
    )

    # HWC -> CHW
    image_tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).permute(2, 0, 1)

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(image_tensor)

        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

    prediction = (
        probability >= THRESHOLD
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    intersection = np.sum(
        prediction * mask
    )

    pred_pixels = prediction.sum()

    union = (
        prediction.sum()
        + mask.sum()
        - intersection
    )

    dice = (
        2.0 * intersection
        /
        (
            prediction.sum()
            + mask.sum()
            + 1e-8
        )
    )

    iou = (
        intersection
        /
        (union + 1e-8)
    )

    true_positive = intersection

    false_positive = np.sum(
        (prediction == 1)
        &
        (mask == 0)
    )

    false_negative = np.sum(
        (prediction == 0)
        &
        (mask == 1)
    )

    precision = (
        true_positive
        /
        (
            true_positive
            + false_positive
            + 1e-8
        )
    )

    recall = (
        true_positive
        /
        (
            true_positive
            + false_negative
            + 1e-8
        )
    )

    results[category]["dice"].append(dice)
    results[category]["iou"].append(iou)
    results[category]["precision"].append(precision)
    results[category]["recall"].append(recall)

    # Completely missed
    if pred_pixels == 0:
        results[category]["missed"] += 1

    # Progress
    if counter % 100 == 0:
        print(
            f"Processed {counter}/{len(val_indices)}"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("LANDSLIDE SIZE ANALYSIS RESULTS")
print("=" * 90)

print(
    f"{'Category':<18}"
    f"{'Patches':>10}"
    f"{'Dice':>12}"
    f"{'IoU':>12}"
    f"{'Precision':>14}"
    f"{'Recall':>12}"
    f"{'Missed':>12}"
)

print("-" * 90)

for category in categories:

    n = len(
        results[category]["dice"]
    )

    if n == 0:
        continue

    avg_dice = np.mean(
        results[category]["dice"]
    )

    avg_iou = np.mean(
        results[category]["iou"]
    )

    avg_precision = np.mean(
        results[category]["precision"]
    )

    avg_recall = np.mean(
        results[category]["recall"]
    )

    missed = results[
        category
    ]["missed"]

    print(
        f"{category:<18}"
        f"{n:>10}"
        f"{avg_dice:>12.4f}"
        f"{avg_iou:>12.4f}"
        f"{avg_precision:>14.4f}"
        f"{avg_recall:>12.4f}"
        f"{missed:>12}"
    )


# ============================================================
# OVERALL
# ============================================================

all_dice = []
all_iou = []
all_precision = []
all_recall = []

for category in results:

    all_dice.extend(
        results[category]["dice"]
    )

    all_iou.extend(
        results[category]["iou"]
    )

    all_precision.extend(
        results[category]["precision"]
    )

    all_recall.extend(
        results[category]["recall"]
    )

print("-" * 90)

print(
    f"{'ALL LANDSLIDES':<18}"
    f"{len(all_dice):>10}"
    f"{np.mean(all_dice):>12.4f}"
    f"{np.mean(all_iou):>12.4f}"
    f"{np.mean(all_precision):>14.4f}"
    f"{np.mean(all_recall):>12.4f}"
)

print("=" * 90)

print("\nAnalysis complete.")