# 🏥 MIRA – Health Prediction App

## Overview

**MIRA** is a Machine Learning-powered health prediction web application built using **Streamlit**.
It analyzes patient blood test parameters (Glucose, Haemoglobin, Cholesterol) and predicts health risk levels using a trained **Random Forest** model. It also provides full patient record management with an integrated SQLite database.



## Key Features

- ML-based health risk prediction (Random Forest)
- Confidence score + reason for every prediction
- Full CRUD — Add, View, Update, Delete patient records
- Search patients by name or email
- Dashboard metrics (high glucose, low haemoglobin, high cholesterol counts)
- Input validation with Regex (email) and range checks
- Clean UI with custom CSS styling

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / Backend | Streamlit |
| Machine Learning | scikit-learn (Random Forest) |
| Data Processing | Pandas, NumPy |
| Database | SQLite (`health.db`) |
| Validation | Python `re` (Regex), `datetime` |

---

## ML Model Details

- **Algorithm:** Random Forest Classifier
- **Input Features:** Glucose (mg/dL), Haemoglobin (g/dL), Cholesterol (mg/dL)
- **Output:** Risk label (High / Low), Confidence %, Reason
- **Training Data:** ~1,400 synthetic samples based on published clinical reference ranges
- **Evaluation:** 5-fold cross-validation + classification report on 20% test split

### Clinical Reference Ranges Used

| Parameter | Normal | Borderline | High Risk |
|---|---|---|---|
| Glucose | 70–99 mg/dL | 100–125 mg/dL | ≥ 126 mg/dL |
| Haemoglobin | 12.0–17.5 g/dL | — | < 12.0 g/dL |
| Cholesterol | < 200 mg/dL | 200–239 mg/dL | ≥ 240 mg/dL |

---

## Project Structure

```
Health_prediction_app/
│
├── app.py                # Main Streamlit application (UI + CRUD + DB)
├── my_model.py           # ML model loader and predict_health() function
├── model_training.py     # Script to train and save model.pkl
├── requirements.txt      # Python dependencies
├── .gitignore            # Excludes db, pkl, pycache files
└── README.md             # Project documentation
```

> `health.db` and `model.pkl` are generated locally and excluded from version control via `.gitignore`.

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/bhakti2701/Health_prediction_app.git
cd Health_prediction_app
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the ML Model

Run this **once** before starting the app. It generates `model.pkl`.

```bash
python model_training.py
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Application Modules

### ➤ Add Patient
- Enter name, date of birth, email, and blood test values
- Get an instant ML prediction with confidence score and reason
- Record is saved to the SQLite database

### ➤ View Patients
- See all patient records in a table
- Dashboard cards: total patients, high glucose, high cholesterol, low haemoglobin
- Search by name or email

### ➤ Update Patient
- Edit any patient's details and test values
- ML prediction is re-run automatically on update

### ➤ Delete Patient
- Permanently remove a patient record with confirmation checkbox

---

## Validation Rules

- Name must not be empty
- Date of birth must be in the past
- Email must match standard format (validated via Regex)
- Glucose, Haemoglobin, Cholesterol must be positive numbers
- Duplicate emails are rejected (unique constraint in DB)
---

## Author

**Bhakti Chaudhari**
Data Analyst | AI/ML Enthusiast

---

*This project demonstrates a complete end-to-end data application with CRUD operations, machine learning prediction, input validation, and database management.*
