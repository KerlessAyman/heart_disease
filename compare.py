from experta import *
from rules import HeartDiseaseExpert
import pandas as pd
import joblib

def get_int_input(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt))     
            if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                print(f"Please enter a value between {min_value} and {max_value}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def get_float_input(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = float(input(prompt))
            if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                print(f"Please enter a value between {min_value} and {max_value}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def get_choice_input(prompt, choices):
    while True:
        value = input(prompt).lower()
        if value in choices:
            return value
        else:
            print(f"Invalid choice. Please enter one of {choices}.")

def get_user_input():
    patient_data = {
        "age": get_int_input("Enter age: ", min_value=0, max_value=120),
        "cholesterol": get_int_input("Enter cholesterol level: ", min_value=0),
        "blood_pressure": get_int_input("Enter blood pressure: ", min_value=0),
        "smoking": get_choice_input("Do you smoke? (yes/no): ", ["yes", "no"]),
        "exercise": get_choice_input("Exercise frequency (none/irregular/regular): ", ["none", "irregular", "regular"]),
        "bmi": get_float_input("Enter BMI: ", min_value=0),
        "glucose": get_int_input("Enter glucose level: ", min_value=0),
        "family_history": get_choice_input("Family history of heart disease? (yes/no): ", ["yes", "no"]),
        "diet": get_choice_input("Diet type (healthy/unhealthy): ", ["healthy", "unhealthy"]),
        "alcohol": get_choice_input("Do you consume alcohol? (yes/no): ", ["yes", "no"]),
        # Add other categorical inputs if your dataset had columns like cp, restecg, ca, thal, etc.
        "cp": get_int_input("Enter chest pain type (1/2/3/4): ", min_value=1, max_value=4),
        "restecg": get_int_input("Enter restecg (0/1/2): ", min_value=0, max_value=2),
        "exang": get_int_input("Exercise induced angina? (0/1): ", min_value=0, max_value=1),
        "oldpeak": get_float_input("Enter oldpeak value: ", min_value=0),
        "slope": get_int_input("Enter slope (1/2/3): ", min_value=1, max_value=3),
        "ca": get_int_input("Enter ca (0/1/2/3/4): ", min_value=0, max_value=4),
        "thal": get_int_input("Enter thal (0/1/2/3): ", min_value=0, max_value=3),
        "fbs": get_int_input("Fasting blood sugar >120? (0/1): ", min_value=0, max_value=1)
    }
    return patient_data

def run_expert_system(patient_data):
    print("\n🧠 Running Expert System Evaluation...")
    engine = HeartDiseaseExpert()
    engine.reset()
    for key, value in patient_data.items():
        engine.declare(Fact(**{key: value}))
    engine.run()

def run_decision_tree_model(patient_data):
    print("\n🤖 Running Decision Tree Prediction...")
    model = joblib.load('heart_disease_decision_tree_model.joblib')
    feature_columns = joblib.load('model_features.joblib')

    # Convert input into DataFrame
    df_input = pd.DataFrame([patient_data])

    # One-hot encode input
    df_input_encoded = pd.get_dummies(df_input)

    # Add missing columns
    for col in feature_columns:
        if col not in df_input_encoded.columns:
            df_input_encoded[col] = 0

    # Ensure correct column order
    df_input_encoded = df_input_encoded[feature_columns]

    prediction = model.predict(df_input_encoded)[0]
    if prediction == 1:
        print("Decision Tree Model Prediction: Likely to have heart disease.")
    else:
        print("Decision Tree Model Prediction: Unlikely to have heart disease.")

if __name__ == "__main__":
    patient_data = get_user_input()
    run_expert_system(patient_data)
    run_decision_tree_model(patient_data)