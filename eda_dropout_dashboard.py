
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder

# Load data and model
df = pd.read_csv("students_dropout_academic_success.csv")
model = joblib.load("student_dropout_predictor_rf.pkl")

# Sidebar
st.sidebar.title("Student Dropout Predictor")
st.sidebar.markdown("This dashboard summarizes the EDA and model performance.")

# Target distribution
st.header("🎯 Target Variable Distribution")
target_dist = df['target'].value_counts()
st.bar_chart(target_dist)

# Feature importances (manually entered from previous analysis)
feature_importance = {
    "Curricular units 2nd sem (approved)": 0.131,
    "Curricular units 2nd sem (grade)": 0.111,
    "Curricular units 1st sem (approved)": 0.097,
    "Curricular units 1st sem (grade)": 0.064,
    "Curricular units 2nd sem (evaluations)": 0.044,
    "Curricular units 1st sem (evaluations)": 0.040,
    "Admission grade": 0.034,
    "Curricular units 2nd sem (enrolled)": 0.031,
    "Curricular units 1st sem (enrolled)": 0.029,
    "Previous qualification (grade)": 0.027
}

st.header("📊 Top 10 Feature Importances")
importance_df = pd.DataFrame(feature_importance.items(), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)
st.dataframe(importance_df)

# ROC Curves (Image from earlier analysis)
st.header("📉 ROC Curve Visualization")
st.image("roc_curve.png")  # Assume saved manually from previous output

# Sample prediction interface
st.header("🤖 Predict Student Outcome")
user_input = {}
for feature in importance_df["Feature"]:
    user_input[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.1)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    label_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
    st.success(f"Predicted Outcome: {label_map[prediction]}")
