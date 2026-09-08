import os
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
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

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
THRESHOLD = 0.50

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("LANDSLIDE U-NET VISUAL ERROR ANALYSIS")
print("=" * 70)
print("Device:", DEVICE)


# ============================================================
# U-NET
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

model = UNet(in_channels=14, out_channels=1)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

# Handle either complete checkpoint or state_dict
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

stats = np.load(STATS_FILE)

means = stats["mean"]
stds = stats["std"]

print("Normalization statistics loaded.")


# ============================================================
# VALIDATION SPLIT
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

split = 3040

train_indices = indices[:split]
val_indices = indices[split:]

print("\nDataset:")
print("Total:", len(indices))
print("Training:", len(train_indices))
print("Validation:", len(val_indices))


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(index):

    img_file = image_files[index]
    mask_file = mask_files[index]

    img_path = os.path.join(IMG_DIR, img_file)
    mask_path = os.path.join(MASK_DIR, mask_file)

    with h5py.File(img_path, "r") as f:
        image = f["img"][:]

    with h5py.File(mask_path, "r") as f:
        mask = f["mask"][:]

    # Normalize
    image = (image - means) / (stds + 1e-8)

    # HWC -> CHW
    image_tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prediction = (probability >= THRESHOLD).astype(np.uint8)

    return image, mask, probability, prediction, img_file


# ============================================================
# RGB DISPLAY
# ============================================================

def create_rgb(image):

    # Landslide4Sense Sentinel-2:
    # B4 = red
    # B3 = green
    # B2 = blue
    #
    # Dataset channel indices:
    # B1 -> 0
    # B2 -> 1
    # B3 -> 2
    # B4 -> 3

    rgb = image[:, :, [3, 2, 1]]

    # Robust percentile stretching
    rgb_min = np.percentile(rgb, 2)
    rgb_max = np.percentile(rgb, 98)

    rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)

    rgb = np.clip(rgb, 0, 1)

    return rgb


# ============================================================
# SELECT EXAMPLES
# ============================================================

# Prefer validation samples containing landslides
landslide_samples = []

for idx in val_indices:

    with h5py.File(
        os.path.join(MASK_DIR, mask_files[idx]),
        "r"
    ) as f:

        mask = f["mask"][:]

    if mask.sum() > 0:
        landslide_samples.append(idx)


random.seed(SEED)

selected = random.sample(
    landslide_samples,
    min(6, len(landslide_samples))
)


# ============================================================
# GENERATE VISUALIZATIONS
# ============================================================

print("\nGenerating visualizations...")

for number, index in enumerate(selected, start=1):

    image, mask, probability, prediction, filename = predict(index)

    rgb = create_rgb(image)

    # Error categories
    true_positive = (prediction == 1) & (mask == 1)
    false_positive = (prediction == 1) & (mask == 0)
    false_negative = (prediction == 0) & (mask == 1)

    dice_intersection = np.sum(prediction * mask)

    dice = (
        2 * dice_intersection /
        (prediction.sum() + mask.sum() + 1e-8)
    )

    print(
        f"{number}. {filename} | "
        f"GT pixels: {mask.sum()} | "
        f"Predicted: {prediction.sum()} | "
        f"Dice: {dice:.4f}"
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title("Satellite RGB")
    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Probability
    axes[2].imshow(probability, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Prediction Probability")
    axes[2].axis("off")

    # Prediction
    axes[3].imshow(prediction, cmap="gray")
    axes[3].set_title(
        f"Prediction (T={THRESHOLD})"
    )
    axes[3].axis("off")

    # Error map
    error_map = np.zeros(
        (mask.shape[0], mask.shape[1], 3)
    )

    # True positive
    error_map[true_positive] = [0, 1, 0]

    # False positive
    error_map[false_positive] = [1, 0, 0]

    # False negative
    error_map[false_negative] = [0, 0, 1]

    axes[4].imshow(error_map)
    axes[4].set_title(
        f"Errors | Dice={dice:.3f}"
    )
    axes[4].axis("off")

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"sample_{number}_{filename.replace('.h5', '')}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print("Visualizations saved to:")
print(OUTPUT_DIR)