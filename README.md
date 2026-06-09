# MIRA – Health Prediction App (Custom ML Model)

Streamlit web app for managing patient blood test records with a
custom-trained RandomForest ML model for health risk prediction.

## Files
| File | Purpose |
|---|---|
| `model_training.py` | Train + save model.pkl — run once |
| `my_model.py` | Loads model.pkl, exposes predict_health() |
| `app.py` | Streamlit CRUD web app |

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (run ONCE)
```bash
python model_training.py
```
This creates `model.pkl` and prints accuracy/F1 scores.

### 3. Run the app
```bash
streamlit run app.py
```
Open http://localhost:8501

## ML Model Details
- **Algorithm**: RandomForestClassifier (100 trees)
- **Features**: Glucose (mg/dL), Haemoglobin (g/dL), Cholesterol (mg/dL)
- **Training data**: 1,400 synthetic samples based on published clinical reference ranges
- **Output**: Risk label + confidence % + reason (e.g. "High Risk (91.2% confidence) — elevated glucose")
- **Preprocessing**: StandardScaler (saved in model.pkl bundle alongside the model)

## Notes
- `model.pkl` and `health.db` are gitignored — never committed to GitHub
- Re-run `model_training.py` any time to retrain with a fresh model
