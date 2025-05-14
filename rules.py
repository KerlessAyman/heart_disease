from experta import *

class HeartDiseaseExpert(KnowledgeEngine):

    @Rule(Fact(cholesterol=P(lambda x: x > 240)) & Fact(age=P(lambda x: x > 50)))
    def high_cholesterol_age_risk(self):
        print("High risk: Cholesterol > 240 and Age > 50")

    @Rule(Fact(blood_pressure=P(lambda x: x > 140)) & Fact(smoking="yes"))
    def high_bp_smoking_risk(self):
        print("High risk: Blood pressure > 140 and Smoking = yes")

    @Rule(Fact(exercise="regular") & Fact(bmi=P(lambda x: x < 25)))
    def low_risk_exercise_bmi(self):
        print("Low risk: Regular exercise and BMI < 25")

    @Rule(Fact(glucose=P(lambda x: x > 150)))
    def high_glucose_risk(self):
        print("Warning: High glucose level detected")

    @Rule(Fact(family_history="yes") & Fact(age=P(lambda x: x > 40)))
    def genetic_age_risk(self):
        print("Increased risk due to family history and age")

    @Rule(Fact(bmi=P(lambda x: x > 30)) & Fact(diet="unhealthy"))
    def obesity_risk(self):
        print("High risk: High BMI and unhealthy diet")

    @Rule(Fact(alcohol="yes") & Fact(blood_pressure=P(lambda x: x > 130)))
    def alcohol_bp_risk(self):
        print("Risk: Alcohol consumption and high blood pressure")

    @Rule(Fact(cholesterol=P(lambda x: x > 240)) & Fact(diet="unhealthy"))
    def cholesterol_diet_risk(self):
        print("Risk: High cholesterol with unhealthy diet")

    @Rule(Fact(exercise="none") & Fact(bmi=P(lambda x: x > 28)))
    def inactivity_risk(self):
        print("High risk due to inactivity and BMI > 28")

    @Rule(Fact(smoking="yes") & Fact(age=P(lambda x: x > 45)))
    def smoking_age_risk(self):
        print("High risk: Smoking with age over 45")