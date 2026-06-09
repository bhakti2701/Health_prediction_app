import re
import sqlite3
from datetime import date

import pandas as pd
import streamlit as st

from my_model import predict_health

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MIRA – Health Prediction",
    page_icon="🏥",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        text-align: center;
    }
    .metric-card .val { font-size: 2rem; font-weight: 700; }
    .metric-card .lbl { font-size: 0.8rem; color: #6c757d; margin-top: 2px; }
    .risk-high { color: #d32f2f; }
    .risk-low  { color: #388e3c; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1565C0;
        border-left: 4px solid #1565C0;
        padding-left: 10px;
        margin: 0.5rem 0 1rem;
    }
    .remark-box {
        background: #f0f7ff;
        border-left: 4px solid #1565C0;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Database ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_connection():
    conn = sqlite3.connect("health.db", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            dob         TEXT NOT NULL,
            email       TEXT NOT NULL UNIQUE,
            glucose     REAL NOT NULL,
            hemoglobin  REAL NOT NULL,
            cholesterol REAL NOT NULL,
            remarks     TEXT
        )
    """)
    conn.commit()
    return conn

conn = get_connection()

# ── Validation ────────────────────────────────────────────────────────────────
def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email)

def validate(name, dob, email, glucose, hemoglobin, cholesterol, exclude_id=None):
    errors = []
    if not name.strip():
        errors.append("Full name is required.")
    if dob >= date.today():
        errors.append("Date of birth cannot be today or a future date.")
    if not is_valid_email(email):
        errors.append("Enter a valid email address (e.g. john@example.com).")
    if glucose <= 0:
        errors.append("Glucose must be a positive number.")
    if hemoglobin <= 0:
        errors.append("Haemoglobin must be a positive number.")
    if cholesterol <= 0:
        errors.append("Cholesterol must be a positive number.")
    q, p = "SELECT id FROM patients WHERE email=?", (email.strip(),)
    if exclude_id:
        q += " AND id!=?"; p += (exclude_id,)
    if conn.execute(q, p).fetchone():
        errors.append("A patient with this email already exists.")
    return errors

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>&#127973; MIRA &mdash; Health Prediction App</h1>
  <p>Medical Intelligence Robotic Automation &nbsp;|&nbsp; ML-Powered Patient Blood Test Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
menu = st.sidebar.radio("", [
    "Add Patient",
    "View Patients",
    "Update Patient",
    "Delete Patient",
], label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption("Reference Ranges")
st.sidebar.markdown("""
| Test | Normal Range |
|---|---|
| Glucose | 70–99 mg/dL |
| Haemoglobin | 12–17.5 g/dL |
| Cholesterol | < 200 mg/dL |
""")
st.sidebar.divider()
st.sidebar.caption("ML Model Info")
st.sidebar.markdown("""
- **Algorithm:** Random Forest  
- **Features:** Glucose, Haemoglobin, Cholesterol  
- **Output:** Risk label + confidence + reason  
- **Training data:** 1,400 synthetic samples based on clinical reference ranges
""")

# ════════════════════════════════════════════════════════════════════════════
# ADD PATIENT
# ════════════════════════════════════════════════════════════════════════════
if menu == "Add Patient":
    st.markdown('<div class="section-header">Add New Patient</div>', unsafe_allow_html=True)

    with st.form("add_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name  = st.text_input("Full Name *", placeholder="e.g. Jane Doe")
            dob   = st.date_input("Date of Birth *",
                                  min_value=date(1900, 1, 1),
                                  max_value=date.today(),
                                  value=date(1990, 1, 1))
            email = st.text_input("Email Address *", placeholder="patient@example.com")
        with col2:
            glucose     = st.number_input("Glucose (mg/dL) *",
                                          min_value=0.0, step=0.1, format="%.1f",
                                          help="Normal fasting: 70–99 mg/dL")
            hemoglobin  = st.number_input("Haemoglobin (g/dL) *",
                                          min_value=0.0, step=0.1, format="%.1f",
                                          help="Normal: 12.0–17.5 g/dL")
            cholesterol = st.number_input("Cholesterol (mg/dL) *",
                                          min_value=0.0, step=0.1, format="%.1f",
                                          help="Desirable: <200 mg/dL")
        submitted = st.form_submit_button("Predict & Save", type="primary", use_container_width=True)

    if submitted:
        errors = validate(name, dob, email, glucose, hemoglobin, cholesterol)
        if errors:
            for e in errors:
                st.error(e)
        else:
            result  = predict_health(glucose, hemoglobin, cholesterol)
            remarks = result["remarks"]

            conn.execute(
                "INSERT INTO patients (name,dob,email,glucose,hemoglobin,cholesterol,remarks) VALUES(?,?,?,?,?,?,?)",
                (name.strip(), str(dob), email.strip(), glucose, hemoglobin, cholesterol, remarks),
            )
            conn.commit()
            st.success("Patient saved successfully!")

            risk_color = "risk-high" if result["label"] == "High Risk" else "risk-low"
            st.markdown(f"""
            <div class="remark-box">
                <b>ML Prediction:</b>
                <span class="{risk_color}"> {result["label"]}</span>
                &nbsp;|&nbsp; <b>Confidence:</b> {result["confidence"]}%
                &nbsp;|&nbsp; <b>Reason:</b> {result["reason"]}
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# VIEW PATIENTS
# ════════════════════════════════════════════════════════════════════════════
elif menu == "View Patients":
    st.markdown('<div class="section-header">All Patient Records</div>', unsafe_allow_html=True)

    rows = conn.execute("SELECT * FROM patients").fetchall()
    if not rows:
        st.info("No patient records yet. Use **Add Patient** from the sidebar.")
    else:
        col_names = ["ID", "Name", "Date of Birth", "Email",
                     "Glucose", "Haemoglobin", "Cholesterol", "Remarks"]
        df = pd.DataFrame(rows, columns=col_names)

        # Stats
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="val">{len(df)}</div><div class="lbl">Total Patients</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="val risk-high">{int((df["Glucose"]>125).sum())}</div><div class="lbl">High Glucose (&gt;125)</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="val risk-high">{int((df["Cholesterol"]>=200).sum())}</div><div class="lbl">High Cholesterol (≥200)</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="val risk-high">{int((df["Haemoglobin"]<12).sum())}</div><div class="lbl">Low Haemoglobin (&lt;12)</div></div>', unsafe_allow_html=True)

        st.divider()
        search = st.text_input("Search by name or email", placeholder="Type to filter...")
        if search:
            mask = (df["Name"].str.contains(search, case=False, na=False) |
                    df["Email"].str.contains(search, case=False, na=False))
            df = df[mask]
            st.caption(f"{len(df)} result(s) found")

        st.dataframe(df, use_container_width=True, hide_index=True,
            column_config={
                "Glucose":     st.column_config.NumberColumn(format="%.1f mg/dL"),
                "Haemoglobin": st.column_config.NumberColumn(format="%.1f g/dL"),
                "Cholesterol": st.column_config.NumberColumn(format="%.1f mg/dL"),
                "Remarks":     st.column_config.TextColumn(width="large"),
            })

# ════════════════════════════════════════════════════════════════════════════
# UPDATE PATIENT
# ════════════════════════════════════════════════════════════════════════════
elif menu == "Update Patient":
    st.markdown('<div class="section-header">Update Patient Record</div>', unsafe_allow_html=True)

    rows = conn.execute("SELECT id, name, email FROM patients").fetchall()
    if not rows:
        st.info("No patients to update.")
    else:
        options = {f"#{r[0]}  {r[1]}  ({r[2]})": r[0] for r in rows}
        chosen  = st.selectbox("Select patient to update", list(options.keys()))
        pid     = options[chosen]

        p = conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
        _, p_name, p_dob, p_email, p_glucose, p_hb, p_chol, p_remarks = p

        if p_remarks:
            st.caption(f"**Current remarks:** {p_remarks}")
        st.divider()

        with st.form("update_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_name  = st.text_input("Full Name *",     value=p_name)
                new_dob   = st.date_input("Date of Birth *",
                                          value=date.fromisoformat(p_dob),
                                          min_value=date(1900, 1, 1),
                                          max_value=date.today())
                new_email = st.text_input("Email Address *", value=p_email)
            with col2:
                new_glucose = st.number_input("Glucose (mg/dL) *",
                                              value=float(p_glucose), min_value=0.0, step=0.1, format="%.1f")
                new_hb      = st.number_input("Haemoglobin (g/dL) *",
                                              value=float(p_hb),      min_value=0.0, step=0.1, format="%.1f")
                new_chol    = st.number_input("Cholesterol (mg/dL) *",
                                              value=float(p_chol),    min_value=0.0, step=0.1, format="%.1f")
            submitted = st.form_submit_button("Update & Re-predict", type="primary", use_container_width=True)

        if submitted:
            errors = validate(new_name, new_dob, new_email, new_glucose, new_hb, new_chol, exclude_id=pid)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                result  = predict_health(new_glucose, new_hb, new_chol)
                remarks = result["remarks"]

                conn.execute(
                    "UPDATE patients SET name=?,dob=?,email=?,glucose=?,hemoglobin=?,cholesterol=?,remarks=? WHERE id=?",
                    (new_name.strip(), str(new_dob), new_email.strip(),
                     new_glucose, new_hb, new_chol, remarks, pid),
                )
                conn.commit()
                st.success("Patient updated successfully!")

                risk_color = "risk-high" if result["label"] == "High Risk" else "risk-low"
                st.markdown(f"""
                <div class="remark-box">
                    <b>New ML Prediction:</b>
                    <span class="{risk_color}"> {result["label"]}</span>
                    &nbsp;|&nbsp; <b>Confidence:</b> {result["confidence"]}%
                    &nbsp;|&nbsp; <b>Reason:</b> {result["reason"]}
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# DELETE PATIENT
# ════════════════════════════════════════════════════════════════════════════
elif menu == "Delete Patient":
    st.markdown('<div class="section-header">Delete Patient Record</div>', unsafe_allow_html=True)

    rows = conn.execute("SELECT id, name, email FROM patients").fetchall()
    if not rows:
        st.info("No patients to delete.")
    else:
        options = {f"#{r[0]}  {r[1]}  ({r[2]})": r[0] for r in rows}
        chosen  = st.selectbox("Select patient to delete", list(options.keys()))
        pid     = options[chosen]

        p = conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
        _, p_name, p_dob, p_email, p_glucose, p_hb, p_chol, p_remarks = p

        with st.expander("Patient details", expanded=True):
            d1, d2, d3 = st.columns(3)
            d1.write(f"**Name:** {p_name}")
            d1.write(f"**DOB:** {p_dob}")
            d2.write(f"**Email:** {p_email}")
            d2.write(f"**Glucose:** {p_glucose} mg/dL")
            d3.write(f"**Haemoglobin:** {p_hb} g/dL")
            d3.write(f"**Cholesterol:** {p_chol} mg/dL")
            if p_remarks:
                st.write(f"**Remarks:** {p_remarks}")

        st.warning("This action is permanent and cannot be undone.")
        confirm = st.checkbox(f"I confirm I want to permanently delete **{p_name}'s** record")

        if st.button("Delete Patient", type="primary", disabled=not confirm):
            conn.execute("DELETE FROM patients WHERE id=?", (pid,))
            conn.commit()
            st.success(f"{p_name}'s record has been deleted.")
            st.rerun()
