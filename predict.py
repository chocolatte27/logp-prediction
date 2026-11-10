"""FastAPI service: predict logP from 8 molecular descriptors."""
import os

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field

FEATURES = [
    "MolWt", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount",
]
MODEL_PATH = os.getenv("MODEL_PATH", "model.ubj")

# Load the model once when the service starts (not on every request)
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

app = FastAPI(title="logP Predictor", version="1.0")


class Molecule(BaseModel):
    # Limits match the data the model was trained on (drug-like molecules)
    MolWt: float = Field(..., gt=0, le=1000, description="Molecular weight (Da)")
    TPSA: float = Field(..., ge=0, description="Topological polar surface area (Å²)")
    NumHDonors: int = Field(..., ge=0, description="H-bond donors")
    NumHAcceptors: int = Field(..., ge=0, description="H-bond acceptors")
    NumRotatableBonds: int = Field(..., ge=0, le=30, description="Rotatable bonds")
    NumAromaticRings: int = Field(..., ge=0, description="Aromatic rings")
    FractionCSP3: float = Field(..., ge=0, le=1, description="Fraction of sp3 carbons")
    HeavyAtomCount: int = Field(..., ge=1, le=80, description="Non-hydrogen atoms")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(mol: Molecule):
    row = pd.DataFrame([mol.model_dump()])[FEATURES]  # keep the column order the model expects
    logp = float(model.predict(row)[0])
    return {
        "predicted_logp": round(logp, 2),
        "lipinski_logp_ok": logp <= 5,  # Lipinski's rule: logP should be 5 or less
    }
