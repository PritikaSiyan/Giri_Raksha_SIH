import os
import h5py
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# MODEL LOCATION
# ============================================================

MODEL_FILE = r"C:\Users\Lenovo\Desktop\Landslide_ai_sih\best_unet.pth"

STATS_FILE = r"C:\Users\Lenovo\Desktop\Landslide_ai_sih\normalization_stats.npz"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

    def __init__(
        self,
        in_channels=14,
        out_channels=1
    ):
        super().__init__()

        self.enc1 = DoubleConv(
            in_channels, 64
        )

        self.enc2 = DoubleConv(
            64, 128
        )

        self.enc3 = DoubleConv(
            128, 256
        )

        self.enc4 = DoubleConv(
            256, 512
        )

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(
            512, 1024
        )

        self.up4 = nn.ConvTranspose2d(
            1024, 512, 2, stride=2
        )

        self.dec4 = DoubleConv(
            1024, 512
        )

        self.up3 = nn.ConvTranspose2d(
            512, 256, 2, stride=2
        )

        self.dec3 = DoubleConv(
            512, 256
        )

        self.up2 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )

        self.dec2 = DoubleConv(
            256, 128
        )

        self.up1 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )

        self.dec1 = DoubleConv(
            128, 64
        )

        self.output = nn.Conv2d(
            64, out_channels, 1
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
# LOAD MODEL
# ============================================================

print("Loading landslide model...")

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


# ============================================================
# LOAD NORMALIZATION
# ============================================================

stats = np.load(
    STATS_FILE
)

MEANS = stats["mean"]
STDS = stats["std"]


print("Model loaded successfully.")
print("Device:", DEVICE)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(image):

    """
    image:
        NumPy array with shape (128, 128, 14)

    returns:
        probability_map
        prediction_mask
        max_probability
        mean_probability
    """

    if image.shape != (128, 128, 14):

        raise ValueError(
            "Expected image shape "
            "(128, 128, 14), got "
            + str(image.shape)
        )

    # Normalize using training statistics
    image = (
        image - MEANS
    ) / (
        STDS + 1e-8
    )

    # HWC -> CHW
    image = torch.tensor(
        image,
        dtype=torch.float32
    ).permute(2, 0, 1)

    # Add batch dimension
    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    # Inference
    with torch.no_grad():

        logits = model(image)

        probabilities = torch.sigmoid(
            logits
        )

    probability_map = (
        probabilities[0, 0]
        .cpu()
        .numpy()
    )

    # Current baseline threshold
    prediction_mask = (
        probability_map >= 0.50
    ).astype(np.uint8)

    max_probability = float(
        probability_map.max()
    )

    mean_probability = float(
        probability_map.mean()
    )

    return (
        probability_map,
        prediction_mask,
        max_probability,
        mean_probability
    )


# ============================================================
# H5 PREDICTION HELPER
# ============================================================

def predict_h5(h5_file):

    """
    Predict directly from a Landslide4Sense H5 image.
    """

    with h5py.File(
        h5_file,
        "r"
    ) as f:

        image = f["img"][:]

    return predict(image)

if __name__ == "__main__":

    test_file = (
        r"C:\Users\Lenovo\Downloads\TrainData"
        r"\img\image_1171.h5"
    )

    (
        probability,
        prediction,
        max_probability,
        mean_probability
    ) = predict_h5(test_file)

    print("\nPrediction test:")
    print(
        "Probability map shape:",
        probability.shape
    )

    print(
        "Prediction mask shape:",
        prediction.shape
    )

    print(
        "Max probability:",
        max_probability
    )

    print(
        "Mean probability:",
        mean_probability
    )

    print(
        "Predicted landslide pixels:",
        int(prediction.sum())
    )