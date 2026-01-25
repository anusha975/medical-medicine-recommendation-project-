from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

# =============================
# Flask App
# =============================
app = Flask(__name__, template_folder="templates", static_folder="static")

# =============================
# BASE DIRECTORY
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =============================
# Load Datasets
# =============================
precautions = pd.read_csv(os.path.join(DATASET_DIR, "precautions_df.csv"))
workout = pd.read_csv(os.path.join(DATASET_DIR, "workout_df.csv"))
description = pd.read_csv(os.path.join(DATASET_DIR, "description.csv"))
medications = pd.read_csv(os.path.join(DATASET_DIR, "medications.csv"))
diets = pd.read_csv(os.path.join(DATASET_DIR, "diets.csv"))

# 🔥 MOST IMPORTANT FILE
training = pd.read_csv(os.path.join(DATASET_DIR, "Training.csv"))

# =============================
# Load Model
# =============================
with open(os.path.join(MODEL_DIR, "svc.pkl"), "rb") as f:
    svc = pickle.load(f)

# =============================
# Extract EXACT features used during training
# =============================
X_COLUMNS = training.drop("prognosis", axis=1).columns.tolist()

print("Model expects features:", len(X_COLUMNS))  # should be 132

# =============================
# Helper Function
# =============================
def helper(disease):
    desc = description[description["Disease"] == disease]["Description"]
    desc = " ".join(desc)

    pre = precautions[precautions["Disease"] == disease][
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    ].values.tolist()

    med = medications[medications["Disease"] == disease]["Medication"].tolist()
    die = diets[diets["Disease"] == disease]["Diet"].tolist()
    wrkout = workout[workout["Disease"] == disease]["workout"].tolist()

    return desc, pre, med, die, wrkout

# =============================
# Prediction Function (CORRECT)
# =============================
def predict_disease(user_symptoms):
    # Create empty dataframe with EXACT training columns
    input_df = pd.DataFrame(
        np.zeros((1, len(X_COLUMNS))),
        columns=X_COLUMNS
    )

    # Fill symptoms
    for symptom in user_symptoms:
        symptom = symptom.strip().lower().replace(" ", "_")
        if symptom in input_df.columns:
            input_df.at[0, symptom] = 1

    prediction = svc.predict(input_df)[0]
    return prediction

# =============================
# Routes
# =============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    symptoms = request.form.get("symptoms")

    if not symptoms:
        return render_template("index.html", message="Please enter valid symptoms")

    user_symptoms = symptoms.split(",")

    disease = predict_disease(user_symptoms)

    desc, precautions_list, meds, diet, workout_list = helper(disease)

    return render_template(
        "index.html",
        predicted_disease=disease,
        dis_des=desc,
        my_precautions=precautions_list[0] if precautions_list else [],
        medications=meds,
        my_diet=diet,
        workout=workout_list
    )

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/developer")
def developer():
    return render_template("developer.html")

@app.route("/blog")
def blog():
    return render_template("blog.html")

# =============================
# Run App
# =============================
if __name__ == "__main__":
    app.run(debug=True)
