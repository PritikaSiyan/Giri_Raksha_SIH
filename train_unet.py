from pathlib import Path
import re
import random

import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(r"C:\Users\Lenovo\Downloads\TrainData")

IMG_DIR = DATA_ROOT / "img"
MASK_DIR = DATA_ROOT / "mask"

STATS_FILE = DATA_ROOT / "normalization_stats.npz"

MODEL_DIR = Path(r"C:\Users\Lenovo\Desktop\Landslide_ai_sih")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "best_unet.pth"

SEED = 42

BATCH_SIZE = 8
NUM_EPOCHS = 30

LEARNING_RATE = 1e-3

NUM_WORKERS = 0

THRESHOLD = 0.5

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


# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("LANDSLIDE4SENSE U-NET TRAINING")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# FIND IMAGE / MASK PAIRS
# ============================================================

def extract_id(filename):
    match = re.search(r"_(\d+)\.h5$", filename)

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

print(f"Total matched samples: {len(pairs)}")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

rng = np.random.default_rng(SEED)

rng.shuffle(pairs)

validation_size = int(
    len(pairs) * 0.20
)

val_pairs = pairs[:validation_size]

train_pairs = pairs[validation_size:]

print(f"Training samples    : {len(train_pairs)}")
print(f"Validation samples  : {len(val_pairs)}")


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

if not STATS_FILE.exists():

    raise FileNotFoundError(
        f"Normalization statistics not found:\n"
        f"{STATS_FILE}\n\n"
        f"Run prepare_dataset.py first."
    )


stats = np.load(STATS_FILE)

mean = stats["mean"].astype(np.float32)
std = stats["std"].astype(np.float32)

print("\nNormalization statistics loaded.")


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
        # IMAGE
        # ----------------------------------------------------

        with h5py.File(image_path, "r") as f:

            image = f["img"][:].astype(
                np.float32
            )


        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        with h5py.File(mask_path, "r") as f:

            mask = f["mask"][:].astype(
                np.float32
            )


        # ----------------------------------------------------
        # NORMALIZATION
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
        # MASK
        # ----------------------------------------------------

        mask = np.expand_dims(
            mask,
            axis=0
        )


        # ----------------------------------------------------
        # TENSORS
        # ----------------------------------------------------

        image = torch.from_numpy(
            image.copy()
        )

        mask = torch.from_numpy(
            mask.copy()
        )


        return image, mask


# ============================================================
# DATASETS
# ============================================================

train_dataset = LandslideDataset(
    train_pairs,
    mean,
    std
)

val_dataset = LandslideDataset(
    val_pairs,
    mean,
    std
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# U-NET BUILDING BLOCK
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

            nn.ReLU(inplace=True),

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

            nn.ReLU(inplace=True)
        )


    def forward(self, x):

        return self.block(x)


# ============================================================
# U-NET
# ============================================================

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
            kernel_size=2,
            stride=2
        )


        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )


        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )


        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )


        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
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
            kernel_size=1
        )


    def forward(self, x):

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        e1 = self.enc1(x)

        p1 = self.pool(e1)


        e2 = self.enc2(p1)

        p2 = self.pool(e2)


        e3 = self.enc3(p2)

        p3 = self.pool(e3)


        e4 = self.enc4(p3)

        p4 = self.pool(e4)


        # ----------------------------------------------------
        # BOTTLENECK
        # ----------------------------------------------------

        b = self.bottleneck(p4)


        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # OUTPUT LOGITS
        # ----------------------------------------------------

        return self.output(d1)


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0
    ):

        super().__init__()

        self.smooth = smooth


    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = probabilities.reshape(
            -1
        )

        targets = targets.reshape(
            -1
        )

        intersection = (
            probabilities * targets
        ).sum()

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            probabilities.sum()
            + targets.sum()
            + self.smooth
        )

        return 1.0 - dice


# ============================================================
# COMBINED LOSS
# ============================================================

class BCEDiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()


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

        return (
            0.5 * bce_loss
            + 0.5 * dice_loss
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


    predictions = predictions.reshape(
        -1
    )

    targets = targets.reshape(
        -1
    )


    true_positive = (
        predictions * targets
    ).sum().item()


    false_positive = (
        predictions * (1 - targets)
    ).sum().item()


    false_negative = (
        (1 - predictions) * targets
    ).sum().item()


    # --------------------------------------------------------
    # DICE
    # --------------------------------------------------------

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    if dice_denominator == 0:

        dice = 1.0

    else:

        dice = (
            2 * true_positive
            / dice_denominator
        )


    # --------------------------------------------------------
    # IOU
    # --------------------------------------------------------

    iou_denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    if iou_denominator == 0:

        iou = 1.0

    else:

        iou = (
            true_positive
            / iou_denominator
        )


    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    precision_denominator = (
        true_positive
        + false_positive
    )

    if precision_denominator == 0:

        precision = 0.0

    else:

        precision = (
            true_positive
            / precision_denominator
        )


    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    recall_denominator = (
        true_positive
        + false_negative
    )

    if recall_denominator == 0:

        recall = 0.0

    else:

        recall = (
            true_positive
            / recall_denominator
        )


    return (
        dice,
        iou,
        precision,
        recall
    )


# ============================================================
# CREATE MODEL
# ============================================================

model = UNet().to(
    DEVICE
)

criterion = BCEDiceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# MODEL SUMMARY
# ============================================================

num_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    f"\nTrainable parameters: "
    f"{num_parameters:,}"
)


# ============================================================
# TRAINING LOOP
# ============================================================

best_dice = -1.0


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0

    for batch_index, (
        images,
        masks
    ) in enumerate(train_loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(
            images
        )


        loss = criterion(
            logits,
            masks
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


    train_loss = (
        running_loss
        / len(train_dataset)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_loss = 0.0

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0


    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )


            logits = model(
                images
            )


            loss = criterion(
                logits,
                masks
            )


            validation_loss += (
                loss.item()
                * images.size(0)
            )


            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            ).float()


            total_tp += (
                predictions * masks
            ).sum().item()


            total_fp += (
                predictions * (1 - masks)
            ).sum().item()


            total_fn += (
                (1 - predictions) * masks
            ).sum().item()


    val_loss = (
        validation_loss
        / len(val_dataset)
    )


    # ========================================================
    # GLOBAL VALIDATION METRICS
    # ========================================================

    dice = (
        2 * total_tp
        / max(
            2 * total_tp
            + total_fp
            + total_fn,
            1e-8
        )
    )


    iou = (
        total_tp
        / max(
            total_tp
            + total_fp
            + total_fn,
            1e-8
        )
    )


    precision = (
        total_tp
        / max(
            total_tp
            + total_fp,
            1e-8
        )
    )


    recall = (
        total_tp
        / max(
            total_tp
            + total_fn,
            1e-8
        )
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"\nEpoch {epoch:02d}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Dice       : {dice:.4f}"
    )

    print(
        f"IoU        : {iou:.4f}"
    )

    print(
        f"Precision  : {precision:.4f}"
    )

    print(
        f"Recall     : {recall:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if dice > best_dice:

        best_dice = dice

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "mean": mean,
            "std": std
        }

        # Make absolutely sure the parent directory exists
        MODEL_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save checkpoint
        torch.save(
            checkpoint,
            str(MODEL_FILE)
        )

    # Verify that the file was actually created
    if MODEL_FILE.exists():

        file_size_mb = (
            MODEL_FILE.stat().st_size
            / (1024 ** 2)
        )

        print(
            f"✓ BEST MODEL SAVED"
        )

        print(
            f"  Path: {MODEL_FILE}"
        )

        print(
            f"  Size: {file_size_mb:.2f} MB"
        )

        print(
            f"  Dice: {best_dice:.4f}"
        )

    else:

        print(
            "❌ ERROR: torch.save() completed "
            "but the model file was not found!"
        )

        print(
            f"Expected location: {MODEL_FILE}"
        )

        raise RuntimeError(
            "Model checkpoint was not created."
        )

# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest validation Dice: "
    f"{best_dice:.4f}"
)

print(
    f"Model saved to:\n"
    f"{MODEL_FILE}"
)