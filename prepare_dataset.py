from pathlib import Path
import re

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(r"C:\Users\Lenovo\Downloads\TrainData")

IMG_DIR = DATA_ROOT / "img"
MASK_DIR = DATA_ROOT / "mask"

RANDOM_SEED = 42
VALIDATION_RATIO = 0.20
BATCH_SIZE = 8

EXPECTED_IMAGE_SHAPE = (128, 128, 14)
EXPECTED_MASK_SHAPE = (128, 128)


# ============================================================
# CHECK DIRECTORIES
# ============================================================

print("=" * 70)
print("LANDSLIDE4SENSE DATASET PREPARATION")
print("=" * 70)

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset root not found:\n{DATA_ROOT}"
    )

if not IMG_DIR.exists():
    raise FileNotFoundError(
        f"Image directory not found:\n{IMG_DIR}"
    )

if not MASK_DIR.exists():
    raise FileNotFoundError(
        f"Mask directory not found:\n{MASK_DIR}"
    )

print(f"\nDataset root : {DATA_ROOT}")
print(f"Image folder : {IMG_DIR}")
print(f"Mask folder  : {MASK_DIR}")


# ============================================================
# HELPER FUNCTION
# ============================================================

def extract_id(filename: str) -> int:
    """
    Extract numerical ID from filenames such as:

        image_123.h5
        mask_123.h5

    Returns:
        123
    """

    match = re.search(r"_(\d+)\.h5$", filename)

    if match is None:
        raise ValueError(
            f"Could not extract numerical ID from filename: {filename}"
        )

    return int(match.group(1))


# ============================================================
# FIND FILES
# ============================================================

image_files = list(IMG_DIR.glob("image_*.h5"))
mask_files = list(MASK_DIR.glob("mask_*.h5"))

print("\n" + "-" * 70)
print("FILE COUNT")
print("-" * 70)

print(f"Images found : {len(image_files)}")
print(f"Masks found  : {len(mask_files)}")


# ============================================================
# CREATE ID -> FILE DICTIONARIES
# ============================================================

image_dict = {}

for path in image_files:
    idx = extract_id(path.name)

    if idx in image_dict:
        raise RuntimeError(
            f"Duplicate image ID found: {idx}"
        )

    image_dict[idx] = path


mask_dict = {}

for path in mask_files:
    idx = extract_id(path.name)

    if idx in mask_dict:
        raise RuntimeError(
            f"Duplicate mask ID found: {idx}"
        )

    mask_dict[idx] = path


# ============================================================
# CHECK IMAGE/MASK MATCHING
# ============================================================

image_ids = set(image_dict.keys())
mask_ids = set(mask_dict.keys())

missing_masks = sorted(image_ids - mask_ids)
missing_images = sorted(mask_ids - image_ids)

if missing_masks:
    print("\nImages missing masks:")
    print(missing_masks[:20])

if missing_images:
    print("\nMasks missing images:")
    print(missing_images[:20])

if missing_masks or missing_images:
    raise RuntimeError(
        "\nImage/mask mismatch detected."
    )


# ============================================================
# CREATE MATCHED PAIRS
# ============================================================

sample_ids = sorted(image_ids)

pairs = [
    (image_dict[idx], mask_dict[idx])
    for idx in sample_ids
]

print(f"\nMatched pairs : {len(pairs)}")

print("\nFirst 5 matched pairs:")

for img_path, mask_path in pairs[:5]:
    print(
        f"  {img_path.name:<20} <--> {mask_path.name}"
    )


# ============================================================
# INSPECT FIRST SAMPLE
# ============================================================

print("\n" + "-" * 70)
print("SAMPLE VALIDATION")
print("-" * 70)

sample_img_path, sample_mask_path = pairs[0]

with h5py.File(sample_img_path, "r") as f:

    if "img" not in f:
        raise KeyError(
            f"'img' dataset not found inside {sample_img_path}"
        )

    sample_image = f["img"][:]


with h5py.File(sample_mask_path, "r") as f:

    if "mask" not in f:
        raise KeyError(
            f"'mask' dataset not found inside {sample_mask_path}"
        )

    sample_mask = f["mask"][:]


print(f"Sample image file : {sample_img_path.name}")
print(f"Sample mask file  : {sample_mask_path.name}")

print(f"Image shape       : {sample_image.shape}")
print(f"Image dtype       : {sample_image.dtype}")

print(f"Mask shape        : {sample_mask.shape}")
print(f"Mask dtype        : {sample_mask.dtype}")

print(f"Mask unique values: {np.unique(sample_mask)}")


if sample_image.shape != EXPECTED_IMAGE_SHAPE:
    raise ValueError(
        f"\nUnexpected image shape: {sample_image.shape}\n"
        f"Expected: {EXPECTED_IMAGE_SHAPE}"
    )


if sample_mask.shape != EXPECTED_MASK_SHAPE:
    raise ValueError(
        f"\nUnexpected mask shape: {sample_mask.shape}\n"
        f"Expected: {EXPECTED_MASK_SHAPE}"
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "-" * 70)
print("TRAIN / VALIDATION SPLIT")
print("-" * 70)

rng = np.random.default_rng(RANDOM_SEED)

shuffled_pairs = pairs.copy()

rng.shuffle(shuffled_pairs)

num_validation = int(
    len(shuffled_pairs) * VALIDATION_RATIO
)

val_pairs = shuffled_pairs[:num_validation]

train_pairs = shuffled_pairs[num_validation:]

print(f"Total samples     : {len(pairs)}")
print(f"Training samples   : {len(train_pairs)}")
print(f"Validation samples : {len(val_pairs)}")


# ============================================================
# CALCULATE TRAINING NORMALIZATION STATISTICS
# ============================================================

print("\n" + "-" * 70)
print("CALCULATING TRAINING-SET NORMALIZATION STATISTICS")
print("-" * 70)

print(
    "\nThis reads the training images one at a time.\n"
    "It may take a few minutes."
)

NUM_CHANNELS = 14

channel_sum = np.zeros(
    NUM_CHANNELS,
    dtype=np.float64
)

channel_sum_sq = np.zeros(
    NUM_CHANNELS,
    dtype=np.float64
)

total_pixels = 0


for i, (img_path, _) in enumerate(train_pairs, start=1):

    with h5py.File(img_path, "r") as f:

        image = f["img"][:].astype(
            np.float64
        )

    # Check shape
    if image.shape != EXPECTED_IMAGE_SHAPE:
        raise ValueError(
            f"\nUnexpected shape in {img_path.name}: "
            f"{image.shape}"
        )

    # Convert:
    # (128, 128, 14)
    #
    # to:
    # (16384, 14)

    pixels = image.reshape(-1, NUM_CHANNELS)

    channel_sum += np.sum(
        pixels,
        axis=0
    )

    channel_sum_sq += np.sum(
        pixels ** 2,
        axis=0
    )

    total_pixels += pixels.shape[0]

    if i % 100 == 0 or i == len(train_pairs):
        print(
            f"Processed {i:4d} / {len(train_pairs)} images"
        )


# ============================================================
# MEAN AND STANDARD DEVIATION
# ============================================================

mean = channel_sum / total_pixels

variance = (
    channel_sum_sq / total_pixels
) - (mean ** 2)

# Protect against tiny floating-point negative values
variance = np.maximum(
    variance,
    1e-12
)

std = np.sqrt(variance)


# ============================================================
# PRINT NORMALIZATION STATISTICS
# ============================================================

print("\n" + "-" * 70)
print("NORMALIZATION STATISTICS")
print("-" * 70)

for channel in range(NUM_CHANNELS):

    print(
        f"Channel {channel:2d} | "
        f"Mean: {mean[channel]:10.6f} | "
        f"Std: {std[channel]:10.6f}"
    )


# ============================================================
# SAVE NORMALIZATION STATISTICS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Lenovo\Desktop\Landslide_ai_sih")
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

stats_path = PROJECT_ROOT / "normalization_stats.npz"

np.savez(
    stats_path,
    mean=mean.astype(np.float32),
    std=std.astype(np.float32)
)

print(
    f"\nNormalization statistics saved to:\n"
    f"{stats_path}"
)


# ============================================================
# PYTORCH DATASET
# ============================================================

class LandslideDataset(Dataset):

    def __init__(
        self,
        pairs,
        mean,
        std
    ):

        self.pairs = pairs

        self.mean = mean.astype(
            np.float32
        )

        self.std = std.astype(
            np.float32
        )

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, index):

        img_path, mask_path = self.pairs[index]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        with h5py.File(img_path, "r") as f:

            image = f["img"][:].astype(
                np.float32
            )

        # ----------------------------------------------------
        # Load mask
        # ----------------------------------------------------

        with h5py.File(mask_path, "r") as f:

            mask = f["mask"][:].astype(
                np.float32
            )

        # ----------------------------------------------------
        # Verify shape
        # ----------------------------------------------------

        if image.shape != EXPECTED_IMAGE_SHAPE:

            raise ValueError(
                f"Invalid image shape in "
                f"{img_path.name}: {image.shape}"
            )

        if mask.shape != EXPECTED_MASK_SHAPE:

            raise ValueError(
                f"Invalid mask shape in "
                f"{mask_path.name}: {mask.shape}"
            )

        # ----------------------------------------------------
        # Normalize image
        #
        # Broadcasting:
        #
        # image shape = (128,128,14)
        # mean  shape = (14,)
        # std   shape = (14,)
        # ----------------------------------------------------

        image = (
            image - self.mean
        ) / self.std

        # ----------------------------------------------------
        # Convert HWC -> CHW
        #
        # Before:
        # (128,128,14)
        #
        # After:
        # (14,128,128)
        # ----------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # Mask:
        #
        # (128,128)
        #
        # -> (1,128,128)
        # ----------------------------------------------------

        mask = np.expand_dims(
            mask,
            axis=0
        )

        # ----------------------------------------------------
        # Convert to PyTorch tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            image.copy()
        )

        mask = torch.from_numpy(
            mask.copy()
        )

        return image, mask


# ============================================================
# CREATE DATASETS
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
# CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# TEST TRAINING DATALOADER
# ============================================================

print("\n" + "-" * 70)
print("TESTING PYTORCH DATALOADER")
print("-" * 70)

images, masks = next(
    iter(train_loader)
)

print(
    f"Image batch shape : {tuple(images.shape)}"
)

print(
    f"Mask batch shape  : {tuple(masks.shape)}"
)

print(
    f"Image dtype       : {images.dtype}"
)

print(
    f"Mask dtype        : {masks.dtype}"
)

print(
    f"Image min         : {images.min().item():.4f}"
)

print(
    f"Image max         : {images.max().item():.4f}"
)

print(
    f"Image mean        : {images.mean().item():.4f}"
)

print(
    f"Image std         : {images.std().item():.4f}"
)

print(
    f"Mask unique values: "
    f"{torch.unique(masks).tolist()}"
)


# ============================================================
# CHECK LANDSLIDE PIXEL BALANCE
# ============================================================

print("\n" + "-" * 70)
print("CHECKING LANDSLIDE PIXEL BALANCE")
print("-" * 70)

landslide_pixels = 0
total_mask_pixels = 0

patches_with_landslide = 0

# We inspect all training masks.
# Masks are small, so this is relatively inexpensive.

for i, (_, mask_path) in enumerate(train_pairs):

    with h5py.File(mask_path, "r") as f:

        mask = f["mask"][:]

    num_landslide = np.sum(
        mask == 1
    )

    landslide_pixels += num_landslide

    total_mask_pixels += mask.size

    if num_landslide > 0:
        patches_with_landslide += 1


landslide_percentage = (
    landslide_pixels
    / total_mask_pixels
) * 100


print(
    f"Total training pixels        : "
    f"{total_mask_pixels:,}"
)

print(
    f"Landslide pixels             : "
    f"{landslide_pixels:,}"
)

print(
    f"Non-landslide pixels         : "
    f"{total_mask_pixels - landslide_pixels:,}"
)

print(
    f"Landslide pixel percentage   : "
    f"{landslide_percentage:.4f}%"
)

print(
    f"Training patches containing "
    f"landslides: "
    f"{patches_with_landslide:,} / "
    f"{len(train_pairs):,}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATA PIPELINE READY")
print("=" * 70)

print(
    f"\nTrain samples : {len(train_dataset)}"
)

print(
    f"Val samples   : {len(val_dataset)}"
)

print(
    f"Input shape   : (14, 128, 128)"
)

print(
    f"Mask shape    : (1, 128, 128)"
)

print(
    f"Batch size    : {BATCH_SIZE}"
)

print(
    "\nEverything is ready for the U-Net training stage."
)