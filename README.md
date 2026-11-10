# logP Prediction from Molecular Descriptors

DataTalks.Club ML Zoomcamp, midterm project.

**Live service:** https://logp-prediction.onrender.com/docs
(Free hosting sleeps after inactivity, so the first request may take about a minute.)

## 1. Problem description

**LogP** measures whether a molecule prefers fat or water. It strongly affects whether a drug can cross cell membranes. Lipinski's Rule of 5 says an oral drug should have logP ≤ 5. Measuring logP in a lab is slow and costly, so chemists want a quick estimate when screening thousands of candidate molecules.

**Goal:** a regression model that predicts logP from 8 simple molecular descriptors. The model is served through a REST API. A chemist or a screening pipeline sends the 8 numbers and gets a predicted logP plus a check against Lipinski's limit of 5.

**Note:** the target is PubChem's computed **XLogP3** value, not lab-measured logP. The model learns to approximate that calculation from a short summary of the molecule.

## 2. Dataset

[426k PubChem Molecules for LogP Prediction]([https://www.kaggle.com/datasets/malikussaid/426k-pubchem-molecules-for-logp-prediction]) (Kaggle). It contains 426,850 molecules that exist in PubChem, ChEMBL and eMolecules.

- **Target:** `logP_target`
- **Features:** `MolWt`, `TPSA`, `NumHDonors`, `NumHAcceptors`, `NumRotatableBonds`, `NumAromaticRings`, `FractionCSP3`, `HeavyAtomCount`
- The dataset is not included in this repo (it is too big). Download `Computed_Molecules.csv` from Kaggle if you want to re-run training.

## 3. Data preparation and EDA

Everything is in `notebook.ipynb`.

- No missing values. Removed 13 duplicate molecules (same InChIKey).
- Dropped the text identifier columns (`InChIKey`, `SMILES`, `Original_InChI`), which are IDs, not measurements.
- **Outliers:** a handful of extreme molecules (logP down to -35, MolWt up to 4975) squashed the plots. I kept only drug-like molecules (logP between -5 and 10, MolWt ≤ 1000, HeavyAtomCount ≤ 80, RotatableBonds ≤ 30). This removed 479 rows (0.1%), leaving 426,358.
- **Split:** 60% train (255,814), 20% validation (85,272), 20% test (85,272).
- **Correlations with logP:** NumAromaticRings +0.43, TPSA -0.36, FractionCSP3 -0.29, NumHDonors -0.28, NumHAcceptors -0.26. No single feature is strong, so combinations matter.
- **Multicollinearity:** MolWt and HeavyAtomCount are 0.97 correlated, and TPSA is strongly correlated with H-donors and H-acceptors.

## 4. Feature importance

| Feature | RF built-in | Permutation (RMSE increase) |
|---|---|---|
| TPSA | 0.284 | 0.761 |
| MolWt | 0.240 | 0.718 |
| FractionCSP3 | 0.098 | 0.278 |
| NumAromaticRings | 0.214 | 0.237 |
| NumHAcceptors | 0.054 | 0.148 |
| NumRotatableBonds | 0.060 | 0.106 |
| NumHDonors | 0.026 | 0.059 |
| HeavyAtomCount | 0.023 | 0.037 |

Both methods agree that **TPSA and MolWt** matter most. MolWt has a weak linear correlation (0.15) but is the 2nd most important feature, because it matters in combination with other features. HeavyAtomCount scores low only because it nearly duplicates MolWt.

## 5. Models and tuning

Models trained on the training set and compared on the validation set. I tuned XGBoost and Random Forest with random search.

| Model | Val RMSE | Val MAE | Val R² |
|---|---|---|---|
| **XGBoost (tuned)** | **0.680** | **0.493** | **0.781** |
| Random Forest (tuned) | 0.704 | 0.512 | 0.765 |
| Random Forest (default) | 0.740 | 0.553 | 0.740 |
| XGBoost (default) | 0.751 | 0.573 | 0.733 |
| Linear Regression | 0.922 | 0.709 | 0.596 |
| Mean guess (baseline) | 1.452 | 1.132 | 0.000 |

- XGBoost tuned: `max_depth=12`, `learning_rate=0.1`, `min_child_weight=5`, `subsample=0.9`, `colsample_bytree=1.0`, `reg_lambda=5`, with early stopping (1076 trees). `max_depth` was by far the most important setting.
- **Selected model: tuned XGBoost.** It has the best scores and a much smaller file than the Random Forest (about 86 MB vs about 1.9 GB).

**Final test-set score (used only once):** RMSE **0.679**, MAE **0.493**, R² **0.777**. This is almost identical to validation, so there is no sign of overfitting.

## 6. Repository structure

| File | Purpose |
|---|---|
| `notebook.ipynb` | Data prep, EDA, feature importance, model training and tuning |
| `train.py` | Trains the final model and saves `model.ubj` and `metrics.json` |
| `predict.py` | FastAPI prediction service |
| `model.ubj` | Trained XGBoost model |
| `metrics.json` | Final validation and test metrics |
| `requirements.txt` | Pinned dependencies |
| `Dockerfile` | Container for the service |
| `screenshots/` | Evidence of the running service |

## 7. How to run

**Retrain (optional).** Needs the CSV from Kaggle:
```bash
pip install -r requirements.txt
python train.py --data Computed_Molecules.csv --model-out model.ubj --metrics-out metrics.json
```

**Run with Docker:**
```bash
docker build -t logp-service .
docker run -p 8000:8000 logp-service
```

**Run without Docker:**
```bash
pip install -r requirements.txt
uvicorn predict:app --port 8000
```

## 8. Using the API

Interactive docs: `http://localhost:8000/docs` (or the live URL above plus `/docs`).

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"MolWt":180.16,"TPSA":63.6,"NumHDonors":1,"NumHAcceptors":3,"NumRotatableBonds":2,"NumAromaticRings":1,"FractionCSP3":0.111,"HeavyAtomCount":13}'
```

Response (aspirin):
```json
{"predicted_logp": 1.9, "lipinski_logp_ok": true}
```

- `GET /health` returns `{"status": "ok"}`.
- Invalid inputs (for example a negative molecular weight) are rejected with HTTP 422.
- Input ranges match the training data (MolWt ≤ 1000, HeavyAtomCount ≤ 80, RotatableBonds ≤ 30).

## 9. Deployment

Deployed on Render (free tier) from this repo's Dockerfile: https://logp-prediction.onrender.com/docs

Screenshots of the container running locally and of the live service are in `screenshots/`.

## 10. Limitations

- The model reaches R² of about 0.78. The target is computed from the full atom-by-atom structure, and 8 summary numbers cannot capture everything.
- It is less accurate for extreme logP values and for molecules far from the typical range. For aspirin it predicts 1.9 while PubChem lists about 1.2, which is roughly one RMSE.
- It only supports drug-like molecules within the ranges listed above.
- It predicts PubChem's XLogP3 estimate, not lab-measured logP.
- Possible improvements: add fingerprint features from the SMILES text (with RDKit), or try LightGBM or CatBoost.
