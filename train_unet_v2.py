import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = r"C:\Users\Lenovo\Downloads\TrainData"
PROJECT_ROOT = r"C:\Users\Lenovo\Desktop\Landslide_ai_sih"

IMG_DIR = os.path.join(DATA_ROOT, "img")
MASK_DIR = os.path.join(DATA_ROOT, "mask")

STATS_FILE = os.path.join(
    PROJECT_ROOT,
    "normalization_stats.npz"
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "best_unet_v2.pth"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

BATCH_SIZE = 8
EPOCHS = 30

LEARNING_RATE = 1e-3

# Probability of selecting a landslide-containing
# training patch through the weighted sampler.
LANDSLIDE_SAMPLE_WEIGHT = 3.0

# Loss weights
BCE_WEIGHT = 0.30
DICE_WEIGHT = 0.40
FOCAL_WEIGHT = 0.30

FOCAL_GAMMA = 2.0

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 70)
print("LANDSLIDE U-NET V2")
print("=" * 70)
print("Device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

class LandslideDataset(Dataset):

    def __init__(
        self,
        indices,
        image_files,
        mask_files,
        means,
        stds
    ):

        self.indices = indices
        self.image_files = image_files
        self.mask_files = mask_files

        self.means = means
        self.stds = stds

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):

        index = self.indices[position]

        img_path = os.path.join(
            IMG_DIR,
            self.image_files[index]
        )

        mask_path = os.path.join(
            MASK_DIR,
            self.mask_files[index]
        )

        with h5py.File(img_path, "r") as f:
            image = f["img"][:]

        with h5py.File(mask_path, "r") as f:
            mask = f["mask"][:]

        # Normalize
        image = (
            image - self.means
        ) / (
            self.stds + 1e-8
        )

        # HWC -> CHW
        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask


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

    def __init__(
        self,
        in_channels=14,
        out_channels=1
    ):

        super().__init__()

        self.enc1 = DoubleConv(
            in_channels,
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

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        self.output = nn.Conv2d(
            64,
            out_channels,
            1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

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
# LOSSES
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.sigmoid(
            logits
        )

        smooth = 1.0

        intersection = (
            probabilities * targets
        ).sum(
            dim=(1, 2, 3)
        )

        denominator = (
            probabilities.sum(
                dim=(1, 2, 3)
            )
            +
            targets.sum(
                dim=(1, 2, 3)
            )
        )

        dice = (
            2.0 * intersection
            + smooth
        ) / (
            denominator
            + smooth
        )

        return (
            1.0 - dice
        ).mean()


class FocalLoss(nn.Module):

    def __init__(
        self,
        gamma=2.0
    ):

        super().__init__()

        self.gamma = gamma

    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.sigmoid(
            logits
        )

        # Probability assigned to the correct class
        pt = torch.where(
            targets == 1,
            probabilities,
            1 - probabilities
        )

        # BCE per pixel
        bce = nn.functional.binary_cross_entropy(
            probabilities,
            targets,
            reduction="none"
        )

        focal_weight = (
            1 - pt
        ) ** self.gamma

        loss = (
            focal_weight * bce
        )

        return loss.mean()


class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.dice = DiceLoss()

        self.focal = FocalLoss(
            gamma=FOCAL_GAMMA
        )

        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits,
        targets
    ):

        bce_loss = self.bce(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        focal_loss = self.focal(
            logits,
            targets
        )

        total = (
            BCE_WEIGHT * bce_loss
            +
            DICE_WEIGHT * dice_loss
            +
            FOCAL_WEIGHT * focal_loss
        )

        return total


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

image_files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.endswith(".h5")
])

mask_files = sorted([
    f for f in os.listdir(MASK_DIR)
    if f.endswith(".h5")
])

assert len(image_files) == len(mask_files)

print(
    "Images:",
    len(image_files)
)

print(
    "Masks:",
    len(mask_files)
)


# ============================================================
# NORMALIZATION
# ============================================================

stats = np.load(
    STATS_FILE
)

means = stats["mean"]
stds = stats["std"]


# ============================================================
# EXACT SAME SPLIT
# ============================================================

indices = list(
    range(len(image_files))
)

random.seed(SEED)

random.shuffle(indices)

split = 3040

train_indices = indices[:split]
val_indices = indices[split:]

print("\nSplit:")

print(
    "Training:",
    len(train_indices)
)

print(
    "Validation:",
    len(val_indices)
)


# ============================================================
# IDENTIFY LANDSLIDE-CONTAINING TRAINING PATCHES
# ============================================================

print(
    "\nAnalyzing training masks..."
)

landslide_flags = []

landslide_patch_count = 0

for index in train_indices:

    mask_path = os.path.join(
        MASK_DIR,
        mask_files[index]
    )

    with h5py.File(
        mask_path,
        "r"
    ) as f:

        mask = f["mask"][:]

    contains_landslide = (
        mask.sum() > 0
    )

    landslide_flags.append(
        contains_landslide
    )

    if contains_landslide:
        landslide_patch_count += 1


print(
    "Training patches containing landslides:",
    landslide_patch_count,
    "/",
    len(train_indices)
)


# ============================================================
# WEIGHTED SAMPLER
# ============================================================

sample_weights = []

for contains_landslide in landslide_flags:

    if contains_landslide:

        sample_weights.append(
            LANDSLIDE_SAMPLE_WEIGHT
        )

    else:

        sample_weights.append(
            1.0
        )


sample_weights = torch.DoubleTensor(
    sample_weights
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_indices),
    replacement=True
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = LandslideDataset(
    train_indices,
    image_files,
    mask_files,
    means,
    stds
)

val_dataset = LandslideDataset(
    val_indices,
    image_files,
    mask_files,
    means,
    stds
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)


# ============================================================
# MODEL / OPTIMIZER / LOSS
# ============================================================

model = UNet(
    in_channels=14,
    out_channels=1
).to(DEVICE)

criterion = CombinedLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    targets,
    threshold=0.5
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    intersection = (
        predictions * targets
    ).sum()

    pred_sum = predictions.sum()
    target_sum = targets.sum()

    union = (
        pred_sum
        +
        target_sum
        -
        intersection
    )

    dice = (
        2 * intersection
        /
        (
            pred_sum
            +
            target_sum
            +
            1e-8
        )
    )

    iou = (
        intersection
        /
        (
            union
            +
            1e-8
        )
    )

    true_positive = intersection

    false_positive = (
        predictions
        *
        (1 - targets)
    ).sum()

    false_negative = (
        (1 - predictions)
        *
        targets
    ).sum()

    precision = (
        true_positive
        /
        (
            true_positive
            +
            false_positive
            +
            1e-8
        )
    )

    recall = (
        true_positive
        /
        (
            true_positive
            +
            false_negative
            +
            1e-8
        )
    )

    return (
        dice.item(),
        iou.item(),
        precision.item(),
        recall.item()
    )


# ============================================================
# TRAINING
# ============================================================

best_dice = 0.0

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            *
            images.size(0)
        )

    train_loss /= len(
        train_loader.dataset
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0

    batches = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                masks
            )

            val_loss += (
                loss.item()
                *
                images.size(0)
            )

            dice, iou, precision, recall = calculate_metrics(
                outputs,
                masks
            )

            total_dice += dice
            total_iou += iou
            total_precision += precision
            total_recall += recall

            batches += 1


    val_loss /= len(
        val_loader.dataset
    )

    avg_dice = (
        total_dice / batches
    )

    avg_iou = (
        total_iou / batches
    )

    avg_precision = (
        total_precision / batches
    )

    avg_recall = (
        total_recall / batches
    )


    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    if avg_dice > best_dice:

        best_dice = avg_dice

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": avg_dice,
                "val_iou": avg_iou,
                "val_precision": avg_precision,
                "val_recall": avg_recall
            },
            MODEL_FILE
        )

        best_text = "  ✓ BEST MODEL SAVED"

    else:

        best_text = ""


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Dice: {avg_dice:.4f} | "
        f"IoU: {avg_iou:.4f} | "
        f"Precision: {avg_precision:.4f} | "
        f"Recall: {avg_recall:.4f}"
        f"{best_text}"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Best validation Dice:",
    f"{best_dice:.4f}"
)

print(
    "Model saved to:"
)

print(
    MODEL_FILE
)

print("=" * 70)