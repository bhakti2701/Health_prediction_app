 # 🏥 MIRA – Health Prediction App

## Overview

**MIRA** is a Machine Learning-powered health prediction web application built using **Streamlit**.
The application analyzes patient blood test parameters and predicts health risk levels using a trained **Random Forest model**. It also allows full patient data management with an integrated database.

---

## Key Features

* ML-based health risk prediction (Random Forest)
* Input parameters:

  * Glucose
  * Haemoglobin
  * Cholesterol
* Displays:
  * Risk level (High / Low)
  * Confidence score
  * Reason for prediction
* Patient record management (CRUD operations)
  * Add patient
  * View records
  * Update details
  * Delete records
* Search functionality for patient records
* Input validation using Regex (email validation)
* Dashboard metrics (high glucose, cholesterol, etc.)
* Clean UI using Streamlit + custom CSS

---

## Tech Stack

### Frontend & Backend

* Streamlit

### Machine Learning

* Random Forest Algorithm
* Pandas
* NumPy

### Database

* SQLite (`health.db`)

### Other Libraries

* Regex (`re`) for validation
* Datetime

---

## Machine Learning Model

* **Algorithm:** Random Forest
* **Input Features:**

  * Glucose
  * Haemoglobin
  * Cholesterol
* **Output:**

  * Risk Label (High / Low)
  * Confidence Score (%)
  * Explanation (Reason)
* **Training Data:**

  * ~1400 synthetic samples based on clinical ranges

---

## Project Structure

```id="proj1"
Health_prediction_app/
│
├── app.py                # Main Streamlit application
├── my_model.py           # ML model prediction logic
├── health.db             # SQLite database
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## Installation & Setup

### 1.Clone Repository

```bash id="c1"
git clone https://github.com/bhakti2701/Health_prediction_app.git
cd Health_prediction_app
```

### 2.Create Virtual Environment

```bash id="c2"
python -m venv venv
venv\Scripts\activate
```

### 3.Install Dependencies

```bash id="c3"
pip install -r requirements.txt
```

### 4️.Run Application

```bash id="c4"
streamlit run app.py
```

---

## Application Modules

### ➤ Add Patient

* Enter patient details
* Input blood test values
* Get ML prediction instantly
* Save record to database

### ➤ View Patients

* Display all patient records
* Dashboard statistics
* Search by name/email

### ➤ Update Patient

* Modify patient details
* Re-run ML prediction

### ➤ Delete Patient

* Remove patient records permanently

---

## Validation

* Email validation implemented using **Regular Expressions (Regex)**
* Prevents duplicate entries using unique email constraint

---

##  Insights Dashboard

* Total patients count
* High glucose cases (>125 mg/dL)
* High cholesterol cases (≥200 mg/dL)
* Low haemoglobin cases (<12 g/dL)

---

##  Future Enhancements

* Deploy on cloud (Streamlit Cloud / AWS)
* Add authentication system
* Improve model with real-world datasets
* Add more health parameters
* Export reports (PDF/Excel)

---

## Disclaimer

This application is for **educational purposes only** and should not be used for medical diagnosis.

---

## 👩‍💻 Author

**Bhakti Chaudhari**

* Data Analyst | AI/ML Enthusiast

---
**Conclusion**

This project demonstrates a complete end-to-end data application with:
CRUD operations
Machine learning prediction
AI integration
Data validation and database management
