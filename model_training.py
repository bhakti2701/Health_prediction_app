"""
model_training.py
-----------------
Run ONCE before starting the app to generate model.pkl:
    python model_training.py

Trains a RandomForest classifier on synthetic data based on
published clinical reference ranges for glucose, haemoglobin,
and cholesterol.
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── 1. Synthetic dataset based on clinical reference ranges ───────────────────
#
#   Glucose      : Normal fasting 70-99  | Pre-diabetic 100-125 | Diabetic 126+  mg/dL
#   Haemoglobin  : Normal 12.0-17.5      | Anaemia <12                            g/dL
#   Cholesterol  : Desirable <200        | Borderline 200-239   | High 240+       mg/dL

rng = np.random.default_rng(seed=42)

def make_samples(n, g_range, hb_range, chol_range, label):
    glucose     = rng.uniform(*g_range,    n) + rng.normal(0, 3,   n)
    hemoglobin  = rng.uniform(*hb_range,   n) + rng.normal(0, 0.3, n)
    cholesterol = rng.uniform(*chol_range, n) + rng.normal(0, 5,   n)
    return np.column_stack([glucose, hemoglobin, cholesterol]), np.full(n, label)

# Low risk
X0, y0 = make_samples(500, (70,  99),  (12.0, 17.5), (130, 199), 0)
X1, y1 = make_samples(150, (90,  110), (11.5, 13.5), (185, 215), 0)

# High risk — diabetic glucose
X2, y2 = make_samples(200, (126, 350), (12.0, 17.5), (130, 199), 1)
# High risk — anaemia
X3, y3 = make_samples(200, (70,  99),  (5.0,  11.9), (130, 199), 1)
# High risk — high cholesterol
X4, y4 = make_samples(200, (70,  99),  (12.0, 17.5), (240, 400), 1)
# High risk — borderline multiple
X5, y5 = make_samples(150, (100, 125), (10.0, 13.0), (200, 239), 1)

X = np.vstack([X0, X1, X2, X3, X4, X5])
y = np.hstack([y0, y1, y2, y3, y4, y5])

print(f"Dataset      : {len(y)} samples")
print(f"Class balance: Low Risk={int((y==0).sum())}  High Risk={int((y==1).sum())}")

# ── 2. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 4. Train ──────────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train_s, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
cv  = cross_val_score(model, X_train_s, y_train, cv=5, scoring="f1")
pred = model.predict(X_test_s)

print(f"\nCross-val F1  : {cv.mean():.3f} (+/- {cv.std():.3f})")
print(f"\nClassification Report:\n{classification_report(y_test, pred, target_names=['Low Risk','High Risk'])}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, pred)}")
print("\nFeature Importances:")
for name, imp in zip(["glucose","hemoglobin","cholesterol"], model.feature_importances_):
    print(f"  {name:12s}: {imp:.3f}")

# ── 6. Save model + scaler bundle ─────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

print("\nmodel.pkl saved successfully.")
