from pathlib import Path
import re

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(r"C:\Users\Lenovo\Downloads\TrainData")

IMG_DIR = DATA_ROOT / "img"
MASK_DIR = DATA_ROOT / "mask"

PROJECT_ROOT = Path(
    r"C:\Users\Lenovo\Desktop\Landslide_ai_sih"
)

MODEL_FILE = PROJECT_ROOT / "best_unet.pth"
STATS_FILE = PROJECT_ROOT / "normalization_stats.npz"

BATCH_SIZE = 8
NUM_WORKERS = 0

SEED = 42
VALIDATION_RATIO = 0.20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("LANDSLIDE4SENSE - THRESHOLD OPTIMIZATION")
print("=" * 70)

print(f"\nDevice       : {DEVICE}")
print(f"Model        : {MODEL_FILE}")
print(f"Statistics   : {STATS_FILE}")


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_FILE}"
    )

if not STATS_FILE.exists():
    raise FileNotFoundError(
        f"\nNormalization statistics not found:\n{STATS_FILE}"
    )


# ============================================================
# REPRODUCE EXACT SAME DATA SPLIT
# ============================================================

def extract_id(filename):
    match = re.search(
        r"_(\d+)\.h5$",
        filename
    )

    if match is None:
        raise ValueError(
            f"Could not extract ID from {filename}"
        )

    return int(match.group(1))


image_files = IMG_DIR.glob("image_*.h5")
mask_files = MASK_DIR.glob("mask_*.h5")


image_dict = {
    extract_id(f.name): f
    for f in image_files
}


mask_dict = {
    extract_id(f.name): f
    for f in mask_files
}


common_ids = sorted(
    set(image_dict.keys()) &
    set(mask_dict.keys())
)


pairs = [
    (image_dict[idx], mask_dict[idx])
    for idx in common_ids
]


print(
    f"\nTotal matched samples : {len(pairs)}"
)


# ------------------------------------------------------------
# IMPORTANT:
# This reproduces the exact randomization used during
# training.
# ------------------------------------------------------------

rng = np.random.default_rng(SEED)

rng.shuffle(pairs)


validation_size = int(
    len(pairs) * VALIDATION_RATIO
)


val_pairs = pairs[:validation_size]

train_pairs = pairs[validation_size:]


print(
    f"Training samples      : {len(train_pairs)}"
)

print(
    f"Validation samples    : {len(val_pairs)}"
)


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

stats = np.load(
    STATS_FILE
)

mean = stats["mean"].astype(
    np.float32
)

std = stats["std"].astype(
    np.float32
)


# ============================================================
# DATASET
# ============================================================

class LandslideDataset(Dataset):

    def __init__(
        self,
        pairs,
        mean,
        std
    ):

        self.pairs = pairs
        self.mean = mean
        self.std = std


    def __len__(self):

        return len(self.pairs)


    def __getitem__(self, index):

        image_path, mask_path = self.pairs[index]


        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        with h5py.File(
            image_path,
            "r"
        ) as f:

            image = f["img"][:].astype(
                np.float32
            )


        # ----------------------------------------------------
        # Load mask
        # ----------------------------------------------------

        with h5py.File(
            mask_path,
            "r"
        ) as f:

            mask = f["mask"][:].astype(
                np.float32
            )


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        image = (
            image - self.mean
        ) / self.std


        # ----------------------------------------------------
        # HWC -> CHW
        # ----------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )


        # ----------------------------------------------------
        # Add mask channel
        # ----------------------------------------------------

        mask = np.expand_dims(
            mask,
            axis=0
        )


        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            image.copy()
        )

        mask = torch.from_numpy(
            mask.copy()
        )


        return image, mask


# ============================================================
# VALIDATION LOADER
# ============================================================

val_dataset = LandslideDataset(
    val_pairs,
    mean,
    std
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )


    def forward(self, x):

        return self.block(x)


class UNet(nn.Module):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        self.enc1 = DoubleConv(
            14,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )


        # ----------------------------------------------------
        # BOTTLENECK
        # ----------------------------------------------------

        self.bottleneck = DoubleConv(
            512,
            1024
        )


        # ----------------------------------------------------
        # POOLING
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(
            2,
            2
        )


        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            2,
            2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )


        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            2,
            2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )


        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            2,
            2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )


        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            2,
            2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        self.output = nn.Conv2d(
            64,
            1,
            1
        )


    def forward(self, x):

        # Encoder

        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)


        # Bottleneck

        b = self.bottleneck(p4)


        # Decoder

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)


        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)


        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)


        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)


        return self.output(d1)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading best U-Net...")

model = UNet().to(DEVICE)


checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print(
    f"Model loaded successfully."
)

print(
    f"Saved checkpoint epoch: "
    f"{checkpoint.get('epoch', 'unknown')}"
)

print(
    f"Saved Dice: "
    f"{checkpoint.get('best_dice', 'unknown')}"
)


# ============================================================
# THRESHOLDS TO TEST
# ============================================================

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]


# ============================================================
# STORAGE
# ============================================================

results = []


# ============================================================
# EVALUATE ALL VALIDATION IMAGES
# ============================================================

print("\nRunning predictions on validation set...")

print(
    f"Validation images: {len(val_dataset)}"
)


# ------------------------------------------------------------
# Instead of storing all predictions, we store the sigmoid
# probabilities and ground-truth masks.
#
# This allows us to test every threshold without running
# the neural network again.
# ------------------------------------------------------------

all_probabilities = []
all_masks = []


with torch.no_grad():

    for batch_index, (
        images,
        masks
    ) in enumerate(val_loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(
            images
        )

        probabilities = torch.sigmoid(
            logits
        )


        all_probabilities.append(
            probabilities.cpu()
        )

        all_masks.append(
            masks
        )


        if (
            (batch_index + 1) % 20 == 0
            or
            (batch_index + 1) == len(val_loader)
        ):

            print(
                f"Processed batch "
                f"{batch_index + 1}/"
                f"{len(val_loader)}"
            )


# ============================================================
# COMBINE PREDICTIONS
# ============================================================

all_probabilities = torch.cat(
    all_probabilities,
    dim=0
)

all_masks = torch.cat(
    all_masks,
    dim=0
)


print(
    "\nPrediction generation complete."
)


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

print(
    f"{'Threshold':<12}"
    f"{'Dice':<12}"
    f"{'IoU':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
)

print("-" * 70)


for threshold in thresholds:

    predictions = (
        all_probabilities >= threshold
    ).float()


    # --------------------------------------------------------
    # Flatten everything
    # --------------------------------------------------------

    pred = predictions.reshape(-1)
    target = all_masks.reshape(-1)


    # --------------------------------------------------------
    # Confusion components
    # --------------------------------------------------------

    tp = (
        pred * target
    ).sum().item()


    fp = (
        pred * (1 - target)
    ).sum().item()


    fn = (
        (1 - pred) * target
    ).sum().item()


    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice = (
        2 * tp
        / max(
            2 * tp + fp + fn,
            1e-8
        )
    )


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou = (
        tp
        / max(
            tp + fp + fn,
            1e-8
        )
    )


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        tp
        / max(
            tp + fp,
            1e-8
        )
    )


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        tp
        / max(
            tp + fn,
            1e-8
        )
    )


    results.append(
        {
            "threshold": threshold,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall
        }
    )


    print(
        f"{threshold:<12.2f}"
        f"{dice:<12.4f}"
        f"{iou:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
    )


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

best_result = max(
    results,
    key=lambda x: x["dice"]
)


print("\n" + "=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"\nThreshold : "
    f"{best_result['threshold']:.2f}"
)

print(
    f"Dice      : "
    f"{best_result['dice']:.4f}"
)

print(
    f"IoU       : "
    f"{best_result['iou']:.4f}"
)

print(
    f"Precision : "
    f"{best_result['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_result['recall']:.4f}"
)


# ============================================================
# COMPARE WITH 0.50
# ============================================================

baseline = next(
    result
    for result in results
    if result["threshold"] == 0.50
)


improvement = (
    best_result["dice"]
    - baseline["dice"]
)


print("\n" + "=" * 70)
print("COMPARISON WITH CURRENT 0.50 THRESHOLD")
print("=" * 70)

print(
    f"\nDice at 0.50 : "
    f"{baseline['dice']:.4f}"
)

print(
    f"Best Dice     : "
    f"{best_result['dice']:.4f}"
)

print(
    f"Improvement   : "
    f"{improvement:+.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    PROJECT_ROOT /
    "threshold_results.txt"
)


with open(
    results_file,
    "w"
) as f:

    f.write(
        "Landslide4Sense Threshold Evaluation\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    for result in results:

        f.write(
            f"Threshold: "
            f"{result['threshold']:.2f} | "
            f"Dice: "
            f"{result['dice']:.4f} | "
            f"IoU: "
            f"{result['iou']:.4f} | "
            f"Precision: "
            f"{result['precision']:.4f} | "
            f"Recall: "
            f"{result['recall']:.4f}\n"
        )


    f.write(
        "\nBEST THRESHOLD\n"
    )

    f.write(
        f"Threshold: "
        f"{best_result['threshold']:.2f}\n"
    )

    f.write(
        f"Dice: "
        f"{best_result['dice']:.4f}\n"
    )

    f.write(
        f"IoU: "
        f"{best_result['iou']:.4f}\n"
    )

    f.write(
        f"Precision: "
        f"{best_result['precision']:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{best_result['recall']:.4f}\n"
    )


print(
    f"\nResults saved to:"
)

print(
    results_file
)

print("\nThreshold evaluation complete.")