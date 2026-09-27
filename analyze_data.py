"""
Analyze the training data to check label encoding
"""

import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "datasets")

print("=" * 60)
print("ANALYZING TRAINING DATA LABELS")
print("=" * 60)

# Check Diabetes data
print("\n1. DIABETES DATASET")
diabetes = pd.read_csv(os.path.join(DATA, "diabetes.csv"))
diabetes.columns = diabetes.columns.str.strip().str.lower()
print(f"   Total samples: {len(diabetes)}")
print(f"   Diabetes values: {diabetes['diabetes'].unique()}")
print(f"   Diabetes distribution:")
print(f"     0 (Healthy):  {(diabetes['diabetes'] == 0).sum()}")
print(f"     1 (Disease):  {(diabetes['diabetes'] == 1).sum()}")

# Check a few samples
print("\n   Sample rows with diabetes=0 (should be healthy):")
healthy = diabetes[diabetes['diabetes'] == 0].head(3)[['age', 'bmi', 'hba1c_level', 'blood_glucose_level', 'diabetes']]
print(healthy.to_string())

print("\n   Sample rows with diabetes=1 (should be disease):")
diseased = diabetes[diabetes['diabetes'] == 1].head(3)[['age', 'bmi', 'hba1c_level', 'blood_glucose_level', 'diabetes']]
print(diseased.to_string())

print("\n   Average BMI for healthy (0): {:.2f}".format(diabetes[diabetes['diabetes'] == 0]['bmi'].mean()))
print(f"   Average BMI for disease (1): {diabetes[diabetes['diabetes'] == 1]['bmi'].mean():.2f}")

print("\n   Average Glucose for healthy (0): {:.2f}".format(diabetes[diabetes['diabetes'] == 0]['blood_glucose_level'].mean()))
print(f"   Average Glucose for disease (1): {diabetes[diabetes['diabetes'] == 1]['blood_glucose_level'].mean():.2f}")

# Check Kidney data
print("\n\n2. KIDNEY DISEASE DATASET")
kidney = pd.read_csv(os.path.join(DATA, "kidney_disease.csv"))
kidney.columns = kidney.columns.str.strip().str.lower()
print(f"   Total samples: {len(kidney)}")
print(f"   Classification values: {kidney['classification'].unique()}")
kidney_class = kidney['classification'].astype(str).str.strip().str.lower()
print(f"   Classification (first 10): {kidney_class.head(10).tolist()}")

print("\n" + "=" * 60)
