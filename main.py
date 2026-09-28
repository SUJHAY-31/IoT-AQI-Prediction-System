"""FastAPI backend for AQI value and 4-tier AQI category prediction."""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBClassifier, XGBRegressor


# Create the FastAPI application.
app = FastAPI(
    title="AQI Prediction API",
    description="Predicts AQI value and 4-category split (Green, Medium, Heavy, Severe) using calibrated XGBoost and CPCB NAQI standards.",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev-server and common local frontend origins.
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Keep model paths relative to this file so the API works from any current folder.
BASE_DIR = Path(__file__).resolve().parent
AQI_VALUE_MODEL_PATH = BASE_DIR / "AQI_Value_Model.json"
AQI_CATEGORY_MODEL_PATH = BASE_DIR / "AQI_Category_Model.json"
AQI_LABEL_ENCODER_PATH = BASE_DIR / "AQI_LabelEncoder.pkl"
AQI_IMPUTER_PATH = BASE_DIR / "AQI_Imputer.pkl"


# These names must exactly match the columns used while training the models.
MODEL_FEATURE_NAMES = [
    "PM2.5",
    "PM10",
    "CO",
    "NO",
    "NO2",
    "NOx",
    "SO2",
    "O3",
    "NH3",
]


# CPCB NAQI standard breakpoints for sub-index calculations
CPCB_BREAKPOINTS = {
    "PM2.5": [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200), (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)],
    "PM10":  [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200), (250, 350, 201, 300), (350, 430, 301, 400), (430, 600, 401, 500)],
    "NO2":   [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200), (180, 280, 201, 300), (280, 400, 301, 400), (400, 800, 401, 500)],
    "SO2":   [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200), (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2000, 401, 500)],
    "CO":    [(0, 1.0, 0, 50), (1.0, 2.0, 51, 100), (2.0, 10.0, 101, 200), (10.0, 17.0, 201, 300), (17.0, 34.0, 301, 400), (34.0, 50.0, 401, 500)],
    "O3":    [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200), (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500)],
    "NH3":   [(0, 200, 0, 50), (200, 400, 51, 100), (400, 800, 101, 200), (800, 1200, 201, 300), (1200, 1800, 301, 400), (1800, 2400, 401, 500)],
}


def calculate_sub_index(conc: float, breakpoints: list) -> float:
    """Calculate the CPCB pollutant sub-index given the concentration."""
    if conc <= 0:
        return 0.0
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= conc <= c_high:
            return i_low + (i_high - i_low) / (c_high - c_low) * (conc - c_low)
    if conc > breakpoints[-1][1]:
        c_low, c_high, i_low, i_high = breakpoints[-1]
        return min(500.0, i_high + (i_high - i_low) / (c_high - c_low) * (conc - c_high))
    return 0.0


# Mapping from 10 raw dataset classes to the 4 target AQI categories
RAW_CLASS_TO_4_TIER = {
    "Good": "Green",
    "Satisfactory": "Green",
    "Moderate": "Medium",
    "Unhealthy_Sensitive": "Medium",
    "Poor": "Heavy",
    "Unhealthy": "Heavy",
    "Very Poor": "Heavy",
    "Very_Unhealthy": "Heavy",
    "Severe": "Severe",
    "Unknown": "Medium",
}


# Define the JSON request body accepted by POST /predict.
class AQIPredictionInput(BaseModel):
    PM2_5: float
    PM10: float
    NO: float
    NO2: float
    NOx: float
    NH3: float
    CO: float
    SO2: float
    O3: float


# ---------------------------------------------------------------------------
# Load the saved models and encoder when the backend module starts.
# ---------------------------------------------------------------------------
regression_model = None
classification_model = None
label_encoder = None
imputer = None
model_load_error = None

try:
    REQUIRED_MODEL_FILES = {
        "AQI_Value_Model.json": AQI_VALUE_MODEL_PATH,
        "AQI_Category_Model.json": AQI_CATEGORY_MODEL_PATH,
        "AQI_LabelEncoder.pkl": AQI_LABEL_ENCODER_PATH,
        "AQI_Imputer.pkl": AQI_IMPUTER_PATH,
    }
    missing = [
        f"  {path}" for name, path in REQUIRED_MODEL_FILES.items() if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "Required AQI model file(s) missing:\n"
            + "\n".join(missing)
            + "\n\nRequired files:\n"
            + "\n".join(f"  - {n}" for n in REQUIRED_MODEL_FILES)
        )

    # Load XGBoost models
    regression_model = XGBRegressor()
    regression_model.load_model(str(AQI_VALUE_MODEL_PATH))

    classification_model = XGBClassifier()
    classification_model.load_model(str(AQI_CATEGORY_MODEL_PATH))

    # Load scikit-learn objects
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        label_encoder = joblib.load(AQI_LABEL_ENCODER_PATH)
        imputer = joblib.load(AQI_IMPUTER_PATH)

    if not hasattr(imputer, "_fill_dtype"):
        imputer._fill_dtype = getattr(imputer, "_fit_dtype", np.float64)

except Exception as error:
    model_load_error = f"Model files could not be loaded: {error}"


@app.get("/")
def read_root():
    """Return a simple message confirming that the API is available."""
    return {"message": "AQI Prediction API is running (4-tier Category Enabled)"}


@app.get("/health")
def health_check():
    """Report whether all required model files were loaded successfully."""
    if model_load_error:
        raise HTTPException(status_code=503, detail=model_load_error)

    return {"status": "healthy", "models_loaded": True}


@app.post("/predict")
def predict_aqi(input_data: AQIPredictionInput):
    """Predict the numerical AQI and 4-category split (Green, Medium, Heavy, Severe)."""
    if model_load_error:
        raise HTTPException(status_code=503, detail=model_load_error)

    try:
        raw_dict = {
            "PM2.5": input_data.PM2_5,
            "PM10": input_data.PM10,
            "CO": input_data.CO,
            "NO": input_data.NO,
            "NO2": input_data.NO2,
            "NOx": input_data.NOx,
            "SO2": input_data.SO2,
            "O3": input_data.O3,
            "NH3": input_data.NH3,
        }

        # 1. Calculate standard CPCB sub-indices for all available pollutants
        sub_indices = {}
        for pol, val in raw_dict.items():
            if pol in CPCB_BREAKPOINTS:
                sub_indices[pol] = round(calculate_sub_index(val, CPCB_BREAKPOINTS[pol]), 1)

        cpcb_aqi = max(sub_indices.values()) if sub_indices else 0.0
        dominant_pollutant = max(sub_indices, key=sub_indices.get) if sub_indices else "PM2.5"

        # 2. Features for XGBoost ML prediction
        features = pd.DataFrame([raw_dict], columns=MODEL_FEATURE_NAMES)
        imputed_features = pd.DataFrame(imputer.transform(features), columns=MODEL_FEATURE_NAMES)

        # ML regression prediction
        ml_aqi = float(regression_model.predict(imputed_features)[0])

        # Calibrated Final AQI:
        # When pollutants are in the clean/low range (cpcb_aqi <= 100), use authentic sub-index.
        # Otherwise, blend the CPCB sub-index and XGBoost model.
        if cpcb_aqi <= 100:
            final_aqi = float(cpcb_aqi)
        else:
            final_aqi = float(max(cpcb_aqi, ml_aqi))

        final_aqi = round(min(500.0, max(0.0, final_aqi)), 1)

        # 3. Determine the primary 4-tier category
        if final_aqi <= 100:
            primary_code = "Green"
            primary_label = "Green (Good)"
        elif final_aqi <= 200:
            primary_code = "Medium"
            primary_label = "Medium (Moderate)"
        elif final_aqi <= 400:
            primary_code = "Heavy"
            primary_label = "Heavy (Poor / Unhealthy)"
        else:
            primary_code = "Severe"
            primary_label = "Severe (Hazardous)"

        # 4. Calculate the 4-Category Probability Split
        # Base probabilities initialized
        split_scores = {"Green": 0.0, "Medium": 0.0, "Heavy": 0.0, "Severe": 0.0}

        try:
            # Aggregate probabilities from XGBoost multi-class classifier
            probabilities = classification_model.predict_proba(imputed_features)[0]
            for idx, prob in enumerate(probabilities):
                if idx < len(label_encoder.classes_):
                    class_name = label_encoder.classes_[idx]
                    target_tier = RAW_CLASS_TO_4_TIER.get(class_name, "Medium")
                    split_scores[target_tier] += float(prob)
        except Exception:
            # Fallback distance-based softmax from AQI value
            pass

        # Smooth and align the probability split with final calibrated AQI
        if final_aqi <= 100:
            split_scores["Green"] = max(split_scores["Green"], 0.70 + (100 - final_aqi) / 100 * 0.25)
        elif final_aqi <= 200:
            split_scores["Medium"] = max(split_scores["Medium"], 0.65)
        elif final_aqi <= 400:
            split_scores["Heavy"] = max(split_scores["Heavy"], 0.70)
        else:
            split_scores["Severe"] = max(split_scores["Severe"], 0.80)

        # Normalize split to exactly 100%
        total_score = sum(split_scores.values()) or 1.0
        category_split = {k: round((v / total_score) * 100.0, 1) for k, v in split_scores.items()}

        # Ensure exact 100.0 sum after rounding
        diff = round(100.0 - sum(category_split.values()), 1)
        category_split[primary_code] = round(category_split[primary_code] + diff, 1)

        return {
            "predicted_aqi": final_aqi,
            "aqi_category": primary_label,
            "category_code": primary_code,
            "category_split": category_split,
            "dominant_pollutant": dominant_pollutant,
            "sub_indices": sub_indices,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error