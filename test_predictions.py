"""
Test script to validate predictions for healthy and high-risk individuals
This helps verify that the models and prediction logic are working correctly
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")

# Load models and scalers
print("=" * 60)
print("LOADING MODELS FOR VALIDATION")
print("=" * 60)

heart_model = joblib.load(os.path.join(MODELS, "heart_model.pkl"))
heart_scaler = joblib.load(os.path.join(MODELS, "heart_scaler.pkl"))

diabetes_model = joblib.load(os.path.join(MODELS, "diabetes_model.pkl"))
diabetes_scaler = joblib.load(os.path.join(MODELS, "diabetes_scaler.pkl"))

kidney_model = joblib.load(os.path.join(MODELS, "kidney_model.pkl"))
kidney_scaler = joblib.load(os.path.join(MODELS, "kidney_scaler.pkl"))

print("✓ All models loaded successfully!")


# ============================================================
# HEALTHY PERSON TEST CASE
# ============================================================
print("\n" + "=" * 60)
print("TEST 1: HEALTHY PERSON (Should give LOW RISK for all)")
print("=" * 60)

# Healthy person: young, good vitals
age = 30
gender = 0  # Female
bp = 115    # Normal
chol = 180  # Good
hr = 70     # Normal
bmi = 21    # Normal
glucose = 90  # Normal

print(f"\nTest Subject: Age {age}, Gender={'Female' if gender == 0 else 'Male'}")
print(f"  BP: {bp}, Cholesterol: {chol}, HR: {hr}")
print(f"  BMI: {bmi}, Glucose: {glucose}")

# Heart Test
print("\n  Heart Disease Prediction:")
heart_features = np.array([[age, gender, bp, chol, hr]])
heart_scaled = heart_scaler.transform(heart_features)
heart_prob = heart_model.predict_proba(heart_scaled)[0]
print(f"    Class 0 (Healthy): {heart_prob[0]:.2%}, Class 1 (Disease): {heart_prob[1]:.2%}")
heart_pred = heart_model.predict(heart_scaled)[0]
print(f"    Prediction: {'Disease' if heart_pred == 1 else 'Healthy'}")

# Diabetes Test
print("\n  Diabetes Prediction:")
hba1c = (0.0915 * glucose + 2.15)
diabetes_features = np.array([[age, bmi, hba1c, glucose]])
diabetes_scaled = diabetes_scaler.transform(diabetes_features)
diabetes_prob = diabetes_model.predict_proba(diabetes_scaled)[0]
print(f"    Class 0 (Healthy): {diabetes_prob[0]:.2%}, Class 1 (Disease): {diabetes_prob[1]:.2%}")
diabetes_pred = diabetes_model.predict(diabetes_scaled)[0]
print(f"    Prediction: {'Disease' if diabetes_pred == 1 else 'Healthy'}")

# Kidney Test
print("\n  Kidney Disease Prediction:")
kidney_features = np.array([[age, bp, glucose]])
kidney_scaled = kidney_scaler.transform(kidney_features)
kidney_prob = kidney_model.predict_proba(kidney_scaled)[0]
print(f"    Class 0 (Healthy): {kidney_prob[0]:.2%}, Class 1 (Disease): {kidney_prob[1]:.2%}")
kidney_pred = kidney_model.predict(kidney_scaled)[0]
print(f"    Prediction: {'Disease' if kidney_pred == 1 else 'Healthy'}")


# ============================================================
# HIGH-RISK PERSON TEST CASE
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: HIGH-RISK PERSON (Should give AT RISK for all)")
print("=" * 60)

# High-risk person: older, poor vitals
age = 65
gender = 1  # Male
bp = 160    # High
chol = 280  # High
hr = 95     # Elevated
bmi = 32    # Overweight
glucose = 180  # High

print(f"\nTest Subject: Age {age}, Gender={'Female' if gender == 0 else 'Male'}")
print(f"  BP: {bp}, Cholesterol: {chol}, HR: {hr}")
print(f"  BMI: {bmi}, Glucose: {glucose}")

# Heart Test
print("\n  Heart Disease Prediction:")
heart_features = np.array([[age, gender, bp, chol, hr]])
heart_scaled = heart_scaler.transform(heart_features)
heart_prob = heart_model.predict_proba(heart_scaled)[0]
print(f"    Class 0 (Healthy): {heart_prob[0]:.2%}, Class 1 (Disease): {heart_prob[1]:.2%}")
heart_pred = heart_model.predict(heart_scaled)[0]
print(f"    Prediction: {'Disease' if heart_pred == 1 else 'Healthy'}")

# Diabetes Test
print("\n  Diabetes Prediction:")
hba1c = (0.0915 * glucose + 2.15)
diabetes_features = np.array([[age, bmi, hba1c, glucose]])
diabetes_scaled = diabetes_scaler.transform(diabetes_features)
diabetes_prob = diabetes_model.predict_proba(diabetes_scaled)[0]
print(f"    Class 0 (Healthy): {diabetes_prob[0]:.2%}, Class 1 (Disease): {diabetes_prob[1]:.2%}")
diabetes_pred = diabetes_model.predict(diabetes_scaled)[0]
print(f"    Prediction: {'Disease' if diabetes_pred == 1 else 'Healthy'}")

# Kidney Test
print("\n  Kidney Disease Prediction:")
kidney_features = np.array([[age, bp, glucose]])
kidney_scaled = kidney_scaler.transform(kidney_features)
kidney_prob = kidney_model.predict_proba(kidney_scaled)[0]
print(f"    Class 0 (Healthy): {kidney_prob[0]:.2%}, Class 1 (Disease): {kidney_prob[1]:.2%}")
kidney_pred = kidney_model.predict(kidney_scaled)[0]
print(f"    Prediction: {'Disease' if kidney_pred == 1 else 'Healthy'}")

print("\n" + "=" * 60)
print("✅ VALIDATION COMPLETE")
print("=" * 60)
print("\nIf Test 1 shows mostly LOW RISK predictions and Test 2")
print("shows mostly HIGH RISK predictions, the models are working correctly!")
print("\nNote:")
print("  - Class 0 = Healthy / No Disease")
print("  - Class 1 = Disease Present")
