import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("=" * 60)
print("STARTING MODEL TRAINING PROCESS")
print("=" * 60)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "datasets")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

print(f"\nBase directory: {BASE}")
print(f"Data directory: {DATA}")
print(f"Models directory: {MODELS}")


def save_model(model, scaler, name):
    print(f"\nSaving {name} model...")
    joblib.dump(model, os.path.join(MODELS, name + "_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS, name + "_scaler.pkl"))
    print(f"✓ {name} model saved successfully!")


def print_model_metrics(y_test, y_pred, name):
    """Print comprehensive model performance metrics"""
    print(f"\n  Performance Metrics for {name}:")
    print(f"    Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"    Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"    Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"    F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    Confusion Matrix:\n      {cm}")


# ============================================================
# 1. HEART DISEASE  -> age, sex, trestbps(bp), chol, thalach(hr)
# ============================================================
print("\n" + "=" * 60)
print("TRAINING HEART DISEASE MODEL")
print("=" * 60)

print("\nLoading heart disease dataset...")
heart = pd.read_csv(os.path.join(DATA, "heart.csv"))
print(f"✓ Heart dataset loaded successfully!")
print(f"  Original shape: {heart.shape}")
print(f"  Columns: {heart.columns.tolist()}")

heart.columns = heart.columns.str.strip().str.lower()
print(f"  Cleaned columns: {heart.columns.tolist()}")

heart_columns = ["age", "sex", "trestbps", "chol", "thalach"]
print(f"\nSelected features for heart disease: {heart_columns}")

print("\nPreprocessing heart disease data...")
x = heart[heart_columns].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(heart["target"], errors="coerce")
heart_data = pd.concat([x, y.rename("target")], axis=1).dropna()
print(f"  After cleaning: {heart_data.shape}")

x = heart_data[heart_columns]
y = heart_data["target"].astype(int)
print(f"  Features shape: {x.shape}")
print(f"  Target shape: {y.shape}")
print(f"  Target distribution - Disease (1): {(y == 1).sum()}, Healthy (0): {(y == 0).sum()}")

print("\nSplitting data into train (80%) and test (20%)...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
print(f"  Training set: {x_train.shape[0]} samples")
print(f"  Testing set: {x_test.shape[0]} samples")

print("\nScaling features...")
heart_scaler = StandardScaler()
x_train = heart_scaler.fit_transform(x_train)
x_test = heart_scaler.transform(x_test)
print("✓ Feature scaling completed!")

print("\nTraining SVC model...")
heart_model = SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale')
heart_model.fit(x_train, y_train)
print("✓ Model training completed!")

heart_pred = heart_model.predict(x_test)
heart_accuracy = accuracy_score(y_test, heart_pred)
print(f"Heart accuracy: {round(heart_accuracy * 100, 2)}%")
print_model_metrics(y_test, heart_pred, "Heart Disease")

save_model(heart_model, heart_scaler, "heart")

# ============================================================
# 2. DIABETES
# Features: age, bmi, HbA1c_level, blood_glucose_level
# ============================================================
print("\n" + "=" * 60)
print("TRAINING DIABETES MODEL")
print("=" * 60)

print("\nLoading diabetes dataset...")
diabetes = pd.read_csv(
    os.path.join(DATA, "diabetes.csv")
)
print(f"✓ Diabetes dataset loaded successfully!")
print(f"  Original shape: {diabetes.shape}")
print(f"  Columns: {diabetes.columns.tolist()}")

# Clean column names
diabetes.columns = (
    diabetes.columns
    .str.strip()
    .str.lower()
)
print(f"  Cleaned columns: {diabetes.columns.tolist()}")

# Features from YOUR diabetes dataset
diabetes_columns = [
    "age",
    "bmi",
    "hba1c_level",
    "blood_glucose_level"
]

# Target
diabetes_target = "diabetes"

print(f"\nSelected features for diabetes: {diabetes_columns}")
print(f"Target column: {diabetes_target}")

# Check required columns
missing_columns = [
    col for col in diabetes_columns + [diabetes_target]
    if col not in diabetes.columns
]

if missing_columns:
    print(f"ERROR: Missing columns: {missing_columns}")
    raise ValueError(
        f"Missing Diabetes columns: {missing_columns}\n"
        f"Available columns: {diabetes.columns.tolist()}"
    )
else:
    print("✓ All required columns are present!")

print("\nPreprocessing diabetes data...")
# Convert feature columns to numeric
for col in diabetes_columns:
    diabetes[col] = pd.to_numeric(
        diabetes[col],
        errors="coerce"
    )

# Convert target to numeric
diabetes[diabetes_target] = pd.to_numeric(
    diabetes[diabetes_target],
    errors="coerce"
)

# Remove rows containing missing values
diabetes = diabetes.dropna(
    subset=diabetes_columns + [diabetes_target]
)
print(f"  After cleaning: {diabetes.shape}")

# Features
x = diabetes[diabetes_columns]

# Target
y = diabetes[diabetes_target].astype(int)
print(f"  Features shape: {x.shape}")
print(f"  Target shape: {y.shape}")
print(f"  Target distribution - Disease (1): {(y == 1).sum()}, Healthy (0): {(y == 0).sum()}")

print("\nSplitting data into train (80%) and test (20%) with stratification...")
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print(f"  Training set: {x_train.shape[0]} samples")
print(f"  Testing set: {x_test.shape[0]} samples")

print("\nScaling features...")
diabetes_scaler = StandardScaler()
x_train = diabetes_scaler.fit_transform(x_train)
x_test = diabetes_scaler.transform(x_test)
print("✓ Feature scaling completed!")

print("\nTraining Logistic Regression model...")
diabetes_model = LogisticRegression(
    max_iter=1000
)

diabetes_model.fit(
    x_train,
    y_train
)
print("✓ Model training completed!")

# Prediction
diabetes_prediction = diabetes_model.predict(
    x_test
)

# Accuracy
diabetes_accuracy = accuracy_score(
    y_test,
    diabetes_prediction
)

print(f"Diabetes accuracy: {round(diabetes_accuracy * 100, 2)}%")
print_model_metrics(y_test, diabetes_prediction, "Diabetes")

# Save model and scaler
save_model(
    diabetes_model,
    diabetes_scaler,
    "diabetes"
)

# ============================================================
# 3. KIDNEY DISEASE -> age, bp, bgr(glucose)
# ============================================================
print("\n" + "=" * 60)
print("TRAINING KIDNEY DISEASE MODEL")
print("=" * 60)

print("\nLoading kidney disease dataset...")
kidney = pd.read_csv(os.path.join(DATA, "kidney_disease.csv"))
print(f"✓ Kidney dataset loaded successfully!")
print(f"  Original shape: {kidney.shape}")
print(f"  Columns: {kidney.columns.tolist()}")

kidney.columns = kidney.columns.str.strip().str.lower()
print(f"  Cleaned columns: {kidney.columns.tolist()}")

kidney_columns = ["age", "bp", "bgr"]
print(f"\nSelected features for kidney disease: {kidney_columns}")

print("\nPreprocessing kidney disease data...")
for col in kidney_columns:
    kidney[col] = pd.to_numeric(kidney[col], errors="coerce")
    kidney[col] = kidney[col].fillna(kidney[col].median())
    print(f"  Processed {col} (median imputation)")

kidney["classification"] = kidney["classification"].astype(str).str.strip().str.lower()
kidney["classification"] = kidney["classification"].apply(lambda v: 1 if v.startswith("ckd") else 0)
kidney = kidney.dropna(subset=kidney_columns + ["classification"])
print(f"  After cleaning: {kidney.shape}")

x = kidney[kidney_columns]
y = kidney["classification"].astype(int)
print(f"  Features shape: {x.shape}")
print(f"  Target shape: {y.shape}")
print(f"  Target distribution - Disease (1): {(y == 1).sum()}, Healthy (0): {(y == 0).sum()}")

print("\nSplitting data into train (80%) and test (20%)...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
print(f"  Training set: {x_train.shape[0]} samples")
print(f"  Testing set: {x_test.shape[0]} samples")

print("\nScaling features...")
kidney_scaler = StandardScaler()
x_train = kidney_scaler.fit_transform(x_train)
x_test = kidney_scaler.transform(x_test)
print("✓ Feature scaling completed!")

print("\nTraining Random Forest model...")
kidney_model = RandomForestClassifier(n_estimators=200, random_state=42)
kidney_model.fit(x_train, y_train)
print("✓ Model training completed!")

kidney_pred = kidney_model.predict(x_test)
kidney_accuracy = accuracy_score(y_test, kidney_pred)
print(f"Kidney accuracy: {round(kidney_accuracy * 100, 2)}%")
print_model_metrics(y_test, kidney_pred, "Kidney Disease")

save_model(kidney_model, kidney_scaler, "kidney")

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\n" + "=" * 50)
print("ALL 3 MODELS RETRAINED — features now match the web form")
print("=" * 50)

print("\nModels saved in the models folder:")
print("\n✓ Heart Model     : heart_model.pkl")
print("  Heart Scaler    : heart_scaler.pkl")
print(f"  Heart Accuracy  : {round(heart_accuracy * 100, 2)}%")

print("\n✓ Diabetes Model  : diabetes_model.pkl")
print("  Diabetes Scaler : diabetes_scaler.pkl")
print(f"  Diabetes Accuracy: {round(diabetes_accuracy * 100, 2)}%")

print("\n✓ Kidney Model    : kidney_model.pkl")
print("  Kidney Scaler   : kidney_scaler.pkl")
print(f"  Kidney Accuracy : {round(kidney_accuracy * 100, 2)}%")

print("\n" + "=" * 60)
print("All models are ready for deployment!")
print("=" * 60)
print("\nTraining completed!")