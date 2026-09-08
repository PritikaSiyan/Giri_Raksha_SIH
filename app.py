from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os

from predictor import predict_h5

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
# ============================================================
# CREATE APP
# ============================================================

app = FastAPI(
    title="Landslide AI API",
    description="AI-powered landslide detection API",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")
# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "service": "Landslide AI",
        "model": "U-Net",
        "input_channels": 14
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Check file type
    if not file.filename.endswith(".h5"):

        raise HTTPException(
            status_code=400,
            detail="Please upload a .h5 Landslide4Sense image file."
        )

    temp_path = None

    try:

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".h5"
        ) as temp:

            temp.write(
                await file.read()
            )

            temp_path = temp.name

        # Run model
        (
            probability_map,
            prediction_mask,
            max_probability,
            mean_probability
        ) = predict_h5(
            temp_path
        )

        # Calculate predicted pixels
        predicted_pixels = int(
            prediction_mask.sum()
        )

        # Simple risk classification
        if predicted_pixels == 0:

            risk_level = "LOW"

        elif max_probability >= 0.90:

            risk_level = "HIGH"

        else:

            risk_level = "MODERATE"

        return {

            "landslide_detected":
                predicted_pixels > 0,

            "risk_level":
                risk_level,

            "max_probability":
                round(
                    max_probability,
                    4
                ),

            "mean_probability":
                round(
                    mean_probability,
                    4
                ),

            "predicted_landslide_pixels":
                predicted_pixels,

            "image_size":
                [128, 128]

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # Delete temporary file
        if (
            temp_path is not None
            and os.path.exists(temp_path)
        ):

            os.remove(temp_path)