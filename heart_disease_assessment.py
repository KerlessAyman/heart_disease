from fuzzy_logic import compute_risk
from utils import validate_input

def get_user_input():
    inputs = {}
    inputs['age'] = int(input("Enter age: "))
    inputs['sex'] = int(input("Enter sex (0 = female, 1 = male): "))
    inputs['cp'] = int(input("Enter chest pain type (0-3): "))
    inputs['trestbps'] = int(input("Enter resting blood pressure: "))
    inputs['chol'] = int(input("Enter cholesterol level: "))
    inputs['fbs'] = int(input("Enter fasting blood sugar (0 = False, 1 = True): "))
    inputs['restecg'] = int(input("Enter resting ECG results (0-2): "))
    inputs['thalach'] = int(input("Enter maximum heart rate achieved: "))
    inputs['exang'] = int(input("Enter exercise-induced angina (0 = No, 1 = Yes): "))
    inputs['oldpeak'] = float(input("Enter oldpeak: "))
    inputs['slope'] = int(input("Enter slope (0-2): "))
    inputs['ca'] = int(input("Enter number of major vessels (0-3): "))
    inputs['thal'] = int(input("Enter thalassemia type (0-3): "))
    return inputs

def main():
    inputs = get_user_input()
    if not validate_input(inputs):
        print("Invalid input values. Please check the ranges.")
        return
    risk_level = compute_risk(**inputs)
    print(f"Heart Disease Risk Level: {risk_level}")

if __name__ == "__main__":
    main()