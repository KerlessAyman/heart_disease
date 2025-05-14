
import sys
import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
from experta import Fact
from rules import HealthRiskExpert  # Import the expert system

# ✅ Function to get user input
def get_user_input():
    try:
        return {
            "cholesterol": int(input("Enter cholesterol level: ").strip() or 0),
            "age": int(input("Enter your age: ").strip() or 0),
            "blood_pressure": int(input("Enter your blood pressure: ").strip() or 0),
            "smoking": input("Do you smoke? (Yes/No): ").strip(),
            "exercise": input("Do you exercise regularly? (Regular/None): ").strip(),
            "bmi": float(input("Enter your BMI: ").strip() or 0),
            "heart_rate": int(input("Enter your heart rate: ").strip() or 0),
            "chest_pain": input("Do you have chest pain? (Yes/No): ").strip(),
            "diet": input("How is your diet? (Healthy/Unhealthy): ").strip(),
        }
    except ValueError:
        print("❌ Invalid input. Please enter correct values.")
        sys.exit(1)
   
# ✅ Main Execution
if __name__ == "__main__":
    user_data = get_user_input()

    engine = HealthRiskExpert()
    engine.reset()
    
    for key, value in user_data.items():
        engine.declare(Fact(**{key: value}))
    
    print("\n🔍 Analyzing your health risks...\n")
    engine.run()