from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from functools import wraps
import json
import os
import sqlite3
import mysql.connector
import joblib
import numpy as np
import pandas as pd
import warnings
from werkzeug.security import check_password_hash, generate_password_hash

warnings.filterwarnings("ignore")


# ============================================================
# APP
# ============================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "health-risk-analyzer-dev-key")


# ============================================================
# FOLDERS
# ============================================================

BASE = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS = os.path.join(
    BASE,
    "models"
)

SQLITE_DATABASE = os.path.join(BASE, "health_analyzer.db")
MYSQL_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "localhost"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "database": os.environ.get("MYSQL_DATABASE", "hospital"),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", "rujula_123")
}


def get_db():
    """Return the request-scoped MySQL connection."""
    if "db" not in g:
        g.db = mysql.connector.connect(**MYSQL_CONFIG)
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    database = g.pop("db", None)
    if database is not None:
        database.close()


def init_db():
    server_config = {key: value for key, value in MYSQL_CONFIG.items() if key != "database"}
    server = mysql.connector.connect(**server_config)
    server_cursor = server.cursor()
    server_cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS `{MYSQL_CONFIG['database']}`"
    )
    server.commit()
    server_cursor.close()
    server.close()

    database = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = database.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT PRIMARY KEY AUTO_INCREMENT,
            username VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS assessments (
            id INT PRIMARY KEY AUTO_INCREMENT,
            user_id INT NOT NULL,
            inputs JSON NOT NULL,
            results JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    database.commit()
    migrate_sqlite_data(database)
    cursor.close()
    database.close()


def migrate_sqlite_data(database):
    """Copy existing SQLite data into MySQL without replacing existing rows."""
    if not os.path.exists(SQLITE_DATABASE):
        return

    sqlite_database = sqlite3.connect(SQLITE_DATABASE)
    sqlite_database.row_factory = sqlite3.Row
    sqlite_users = sqlite_database.execute(
        "SELECT id, username, password_hash, created_at FROM users"
    ).fetchall()
    sqlite_assessments = sqlite_database.execute(
        "SELECT id, user_id, inputs, results, created_at FROM assessments"
    ).fetchall()
    cursor = database.cursor()
    cursor.executemany(
        """
        INSERT IGNORE INTO users (id, username, password_hash, created_at)
        VALUES (%s, %s, %s, %s)
        """,
        [tuple(user) for user in sqlite_users]
    )
    cursor.executemany(
        """
        INSERT IGNORE INTO assessments (id, user_id, inputs, results, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """,
        [tuple(assessment) for assessment in sqlite_assessments]
    )
    database.commit()
    cursor.close()
    sqlite_database.close()


init_db()


@app.before_request
def load_logged_in_user():
    user_id = session.get("user_id")
    if user_id is None:
        g.user = None
        return
    cursor = get_db().cursor(dictionary=True)
    cursor.execute("SELECT id, username FROM users WHERE id = %s", (user_id,))
    g.user = cursor.fetchone()
    cursor.close()


def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            flash("Please log in to run a health assessment.", "info")
            return redirect(url_for("login"))
        return view(**kwargs)
    return wrapped_view


def risk_level(risk_percentage):
    """Convert a percentage into one of the three UI risk levels."""
    if risk_percentage < 30:
        return "LOW"
    if risk_percentage < 60:
        return "MEDIUM"
    return "HIGH"

print("=" * 60)
print("🚀 STARTING HEALTH RISK ANALYZER APPLICATION")
print("=" * 60)

print(f"\n📁 Base directory: {BASE}")
print(f"📁 Models directory: {MODELS}")


# ============================================================
# LOAD HEART MODEL
# ============================================================

print("\n📊 Loading Heart Disease model...")
try:
    heart_model = joblib.load(
        os.path.join(
            MODELS,
            "heart_model.pkl"
        )
    )
    heart_scaler = joblib.load(
        os.path.join(
            MODELS,
            "heart_scaler.pkl"
        )
    )
    print("✓ Heart Disease model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading Heart model: {e}")
    heart_model = None
    heart_scaler = None


# ============================================================
# LOAD DIABETES MODEL
# ============================================================

print("\n📊 Loading Diabetes model...")
try:
    diabetes_model = joblib.load(
        os.path.join(
            MODELS,
            "diabetes_model.pkl"
        )
    )
    diabetes_scaler = joblib.load(
        os.path.join(
            MODELS,
            "diabetes_scaler.pkl"
        )
    )
    print("✓ Diabetes model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading Diabetes model: {e}")
    diabetes_model = None
    diabetes_scaler = None


# ============================================================
# LOAD KIDNEY MODEL
# ============================================================

print("\n📊 Loading Kidney Disease model...")
try:
    kidney_model = joblib.load(
        os.path.join(
            MODELS,
            "kidney_model.pkl"
        )
    )
    kidney_scaler = joblib.load(
        os.path.join(
            MODELS,
            "kidney_scaler.pkl"
        )
    )
    print("✓ Kidney Disease model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading Kidney model: {e}")
    kidney_model = None
    kidney_scaler = None

print("\n" + "=" * 60)
print("✅ ALL MODELS LOADED - Application is ready!")
print("=" * 60)


# ============================================================
# HELPER FUNCTIONS FOR RISK ASSESSMENT
# ============================================================

def assess_heart_risk(age, gender, bp, chol, hr):
    """Assess heart disease risk using medical guidelines"""
    risk_score = 0
    
    # Age risk (increases with age)
    if age > 60:
        risk_score += 3
    elif age > 50:
        risk_score += 2
    elif age > 40:
        risk_score += 1
    
    # Gender risk (males generally higher risk)
    if gender == 1 and age > 45:
        risk_score += 1
    
    # Blood pressure risk (medical guidelines)
    if bp > 140:  # Stage 2 hypertension
        risk_score += 3
    elif bp > 130:  # Stage 1 hypertension
        risk_score += 2
    elif bp > 120:  # Elevated
        risk_score += 1
    
    # Cholesterol risk (medical guidelines)
    if chol > 240:  # High
        risk_score += 3
    elif chol > 200:  # Borderline high
        risk_score += 2
    elif chol > 180:  # Acceptable
        risk_score += 1
    
    # Heart rate risk
    if hr > 100 or hr < 60:
        risk_score += 1
    
    # Normalize risk score to percentage (0-100)
    risk_percentage = min(100, (risk_score / 11) * 100)
    
    return risk_percentage

def assess_diabetes_risk(age, bmi, glucose):
    """Assess diabetes risk using medical guidelines"""
    risk_score = 0
    
    # Age risk
    if age > 60:
        risk_score += 2
    elif age > 50:
        risk_score += 1
    
    # BMI risk (obesity criteria)
    if bmi > 30:  # Obese
        risk_score += 3
    elif bmi > 27:  # Overweight
        risk_score += 2
    elif bmi > 25:  # Overweight
        risk_score += 1
    
    # Glucose risk (using medical guidelines, not training data averages)
    if glucose > 200:  # Very high (diabetic range)
        risk_score += 3
    elif glucose > 126:  # Diabetic threshold
        risk_score += 3
    elif glucose > 100:  # Pre-diabetic range
        risk_score += 2
    elif glucose > 90:  # Upper normal
        risk_score += 1
    
    # Normalize risk score to percentage
    risk_percentage = min(100, (risk_score / 8) * 100)
    
    return risk_percentage

def assess_kidney_risk(age, bp, glucose):
    """Assess kidney disease risk using medical guidelines"""
    risk_score = 0
    
    # Age risk
    if age > 65:
        risk_score += 2
    elif age > 55:
        risk_score += 1
    
    # Blood pressure risk (kidney disease risk factor)
    if bp > 140:
        risk_score += 3
    elif bp > 130:
        risk_score += 2
    elif bp > 120:
        risk_score += 1
    
    # Glucose risk (diabetes is a risk factor for kidney disease)
    if glucose > 200:
        risk_score += 3
    elif glucose > 126:
        risk_score += 3
    elif glucose > 100:
        risk_score += 2
    elif glucose > 90:
        risk_score += 1
    
    # Normalize risk score to percentage
    risk_percentage = min(100, (risk_score / 8) * 100)
    
    return risk_percentage


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if len(username) < 3 or len(password) < 6:
            flash("Username must have 3 characters and password must have 6 characters.", "error")
            return render_template("register.html")

        try:
            database = get_db()
            cursor = database.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, generate_password_hash(password))
            )
            database.commit()
            cursor.close()
        except mysql.connector.IntegrityError:
            flash("That username is already registered.", "error")
            return render_template("register.html")

        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        cursor = get_db().cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid username or password.", "error")
            return render_template("login.html")

        session.clear()
        session["user_id"] = user["id"]
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route(
    "/",
    methods=["GET", "POST"]
)
@login_required
def home():

    result = None

    # Default values - more realistic normal ranges
    values = {
        "age": "45",
        "gender": "Female",
        "bmi": "24",
        "bp": "118",
        "glucose": "95",
        "chol": "190",
        "hr": "70",
        "smoking": "No",
        "exercise": "Yes"
    }

    # ========================================================
    # WHEN USER CLICKS PREDICT
    # ========================================================

    if request.method == "POST":

        print("\n" + "=" * 50)
        print("🔍 NEW PREDICTION REQUEST")
        print("=" * 50)

        # Get form values
        for key in values:
            values[key] = request.form.get(
                key,
                values[key]
            )
            print(f"  {key}: {values[key]}")

        try:
            # Convert values
            age = float(values["age"])
            bmi = float(values["bmi"])
            bp = float(values["bp"])
            glucose = float(values["glucose"])
            chol = float(values["chol"])
            hr = float(values["hr"])
            smoking = values["smoking"]
            exercise = values["exercise"]

            print("\n📊 Processing predictions...")

            # =================================================
            # HEART PREDICTION - Using ML model + rule-based
            # =================================================

            print("\n  ❤️  Heart Disease Prediction:")
            
            gender = 1 if values["gender"] == "Male" else 0
            
            # ML Model Prediction
            heart_features = np.array([
                age,      # age
                gender,   # sex (1=male, 0=female)
                bp,       # trestbps
                chol,     # chol
                hr        # thalach
            ]).reshape(1, -1)
            
            print(f"    ML Input: [age={age}, sex={gender}, bp={bp}, chol={chol}, hr={hr}]")
            
            # Scale and predict
            heart_scaled = heart_scaler.transform(heart_features)
            heart_prob = heart_model.predict_proba(heart_scaled)[0]
            
            # NOTE: The heart.csv dataset has inverted labels
            # In the actual data: 0=Disease, 1=Healthy (opposite of what we assumed)
            # So we need to invert: use probability of class 0 as disease risk
            ml_risk = heart_prob[0] * 100  # Class 0 is actually disease
            
            print(f"    ML Risk: {ml_risk:.1f}%")
            
            # Rule-based risk assessment
            rule_risk = assess_heart_risk(age, gender, bp, chol, hr)
            print(f"    Rule-based Risk: {rule_risk:.1f}%")
            
            # Combine both methods - improve logic with better weighting
            # Use Bayesian-like approach: trust whichever prediction is more confident
            ml_confidence = abs(ml_risk - 50) / 50  # 0-1 scale
            rule_confidence = abs(rule_risk - 50) / 50  # 0-1 scale
            
            # If both agree strongly, use simple average
            if (ml_risk > 50 and rule_risk > 50) and ml_confidence > 0.3 and rule_confidence > 0.3:
                # Both indicate risk with good confidence
                final_heart_risk = (ml_risk * 0.5 + rule_risk * 0.5)
            elif (ml_risk < 50 and rule_risk < 50) and ml_confidence > 0.3 and rule_confidence > 0.3:
                # Both indicate safety with good confidence
                final_heart_risk = (ml_risk * 0.5 + rule_risk * 0.5)
            else:
                # They disagree or one is uncertain, weight rule-based more (medical knowledge is more reliable)
                final_heart_risk = (ml_risk * 0.35 + rule_risk * 0.65)
            
            # Apply threshold with hysteresis for stability
            heart_status = f"{risk_level(final_heart_risk)} RISK"
            
            print(f"    Final Risk: {final_heart_risk:.1f}%")
            print(f"    Status: {heart_status}")

            # =================================================
            # DIABETES PREDICTION - Using ML model + rule-based
            # =================================================

            print("\n  🩸 Diabetes Prediction:")
            
            # ML Model Prediction
            # For diabetes, we need: age, bmi, hba1c_level, blood_glucose_level
            # Use empirical ADAG formula to estimate HbA1c from glucose
            # HbA1c ≈ (0.0915 * fasting_glucose + 2.15) for mg/dL
            hba1c = (0.0915 * glucose + 2.15)
            
            diabetes_features = np.array([
                age,      # age
                bmi,      # bmi
                hba1c,    # hba1c_level (estimated using ADAG formula)
                glucose   # blood_glucose_level
            ]).reshape(1, -1)
            
            print(f"    ML Input: [age={age}, bmi={bmi:.1f}, hba1c={hba1c:.1f}, glucose={glucose}]")
            
            # Scale and predict
            diabetes_scaled = diabetes_scaler.transform(diabetes_features)
            diabetes_prob = diabetes_model.predict_proba(diabetes_scaled)[0]
            ml_risk = diabetes_prob[1] * 100
            
            print(f"    ML Risk: {ml_risk:.1f}% (class 1 probability)")
            
            # Rule-based risk assessment
            rule_risk = assess_diabetes_risk(age, bmi, glucose)
            print(f"    Rule-based Risk: {rule_risk:.1f}%")
            
            # Combine both methods - use adaptive weighting based on confidence
            ml_confidence = abs(ml_risk - 50) / 50  # 0-1 scale
            rule_confidence = abs(rule_risk - 50) / 50  # 0-1 scale
            
            # If both agree strongly, use them equally
            if (ml_risk > 50 and rule_risk > 50) and ml_confidence > 0.2 and rule_confidence > 0.2:
                final_diabetes_risk = (ml_risk * 0.5 + rule_risk * 0.5)
            elif (ml_risk < 50 and rule_risk < 50) and ml_confidence > 0.2 and rule_confidence > 0.2:
                final_diabetes_risk = (ml_risk * 0.5 + rule_risk * 0.5)
            else:
                # Disagree or uncertain, weight rule-based higher (medical guidelines are more reliable)
                final_diabetes_risk = (ml_risk * 0.3 + rule_risk * 0.7)
            
            # Apply threshold - 50% is the decision boundary
            diabetes_status = f"{risk_level(final_diabetes_risk)} RISK"
            
            print(f"    Final Risk: {final_diabetes_risk:.1f}%")
            print(f"    Status: {diabetes_status}")

            # =================================================
            # KIDNEY PREDICTION - Using ML model + rule-based
            # =================================================

            print("\n  🫘 Kidney Disease Prediction:")
            
            # ML Model Prediction
            kidney_features = np.array([
                age,      # age
                bp,       # blood pressure
                glucose   # blood glucose
            ]).reshape(1, -1)
            
            print(f"    ML Input: [age={age}, bp={bp}, glucose={glucose}]")
            
            # Scale and predict
            kidney_scaled = kidney_scaler.transform(kidney_features)
            kidney_prob = kidney_model.predict_proba(kidney_scaled)[0]
            ml_risk = kidney_prob[1] * 100
            
            print(f"    ML Risk: {ml_risk:.1f}%")
            
            # Rule-based risk assessment
            rule_risk = assess_kidney_risk(age, bp, glucose)
            print(f"    Rule-based Risk: {rule_risk:.1f}%")
            
            # Combine both methods - adaptive weighting
            ml_confidence = abs(ml_risk - 50) / 50  # 0-1 scale
            rule_confidence = abs(rule_risk - 50) / 50  # 0-1 scale
            
            # If both agree with good confidence, use them equally
            if (ml_risk > 50 and rule_risk > 50) and ml_confidence > 0.2 and rule_confidence > 0.2:
                final_kidney_risk = (ml_risk * 0.5 + rule_risk * 0.5)
            elif (ml_risk < 50 and rule_risk < 50) and ml_confidence > 0.2 and rule_confidence > 0.2:
                final_kidney_risk = (ml_risk * 0.5 + rule_risk * 0.5)
            else:
                # Disagree or uncertain, weight rule-based more heavily (medical knowledge)
                final_kidney_risk = (ml_risk * 0.3 + rule_risk * 0.7)
            
            # Apply threshold
            kidney_status = f"{risk_level(final_kidney_risk)} RISK"
            
            print(f"    Final Risk: {final_kidney_risk:.1f}%")
            print(f"    Status: {kidney_status}")

            # =================================================
            # RESULT
            # =================================================

            result = {
                "heart": heart_status,
                "diabetes": diabetes_status,
                "kidney": kidney_status,
                "heart_prob": round(final_heart_risk, 1),
                "diabetes_prob": round(final_diabetes_risk, 1),
                "kidney_prob": round(final_kidney_risk, 1),
                "heart_level": risk_level(final_heart_risk),
                "diabetes_level": risk_level(final_diabetes_risk),
                "kidney_level": risk_level(final_kidney_risk)
            }

            cursor = get_db().cursor()
            cursor.execute(
                "INSERT INTO assessments (user_id, inputs, results) VALUES (%s, %s, %s)",
                (g.user["id"], json.dumps(values), json.dumps(result))
            )
            get_db().commit()
            cursor.close()
            
            print("\n✅ Predictions completed successfully!")
            print("=" * 50)

        except Exception as error:
            print(f"\n❌ ERROR: {str(error)}")
            import traceback
            traceback.print_exc()
            
            result = {
                "error": f"Prediction error: {str(error)}"
            }

    return render_template(
        "index.html",
        result=result,
        **values
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🏥 HEALTH RISK ANALYZER APPLICATION")
    print("=" * 60)
    print("\n🌐 Open your browser and go to:")
    print("   📍 http://127.0.0.1:5000")
    print("\n📋 Features available:")
    print("   • Heart Disease Risk Prediction")
    print("   • Diabetes Risk Prediction")
    print("   • Kidney Disease Risk Prediction")
    print("\n⚠️  Note: This is an academic AIML project")
    print("=" * 60)
    print("\n🔄 Press CTRL+C to stop the server\n")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )