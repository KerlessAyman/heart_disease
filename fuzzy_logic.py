import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# تعريف جميع المتغيرات الداخلة
age = ctrl.Antecedent(np.arange(0, 101, 1), 'age')
sex = ctrl.Antecedent(np.arange(0, 2, 1), 'sex')
cp = ctrl.Antecedent(np.arange(0, 4, 1), 'cp')
trestbps = ctrl.Antecedent(np.arange(90, 201, 1), 'trestbps')
chol = ctrl.Antecedent(np.arange(100, 601, 1), 'chol')
thalach = ctrl.Antecedent(np.arange(60, 202, 1), 'thalach')
oldpeak = ctrl.Antecedent(np.arange(0, 7, 0.1), 'oldpeak')
slope = ctrl.Antecedent(np.arange(0, 3, 1), 'slope')  # تمت إضافة slope

# متغير الخرج (الخطورة)
risk = ctrl.Consequent(np.arange(0, 11, 1), 'risk')

def define_membership_functions():
    # العمر: شاب، متوسط، كبير
    age.automf(3, names=['Young', 'Middle', 'Old'])
    
    # الجنس: أنثى، ذكر
    sex['Female'] = fuzz.trimf(sex.universe, [0, 0, 1])
    sex['Male'] = fuzz.trimf(sex.universe, [0, 1, 1])
    
    # نوع ألم الصدر: نموذجي، غير نموذجي، غير قلبي، بدون أعراض
    cp.automf(4, names=['Typical', 'Atypical', 'Non-anginal', 'Asymptomatic'])
    
    # ضغط الدم: منخفض، طبيعي، مرتفع
    trestbps.automf(3, names=['Low', 'Normal', 'High'])
    
    # الكوليسترول: منخفض، متوسط، مرتفع
    chol['Low'] = fuzz.gaussmf(chol.universe, 150, 30)
    chol['Medium'] = fuzz.gaussmf(chol.universe, 250, 30)
    chol['High'] = fuzz.gaussmf(chol.universe, 350, 30)
    
    # معدل ضربات القلب: منخفض، متوسط، مرتفع
    thalach.automf(3, names=['Low', 'Medium', 'High'])
    
    # انخفاض ST: منخفض، متوسط، مرتفع
    oldpeak['Low'] = fuzz.trimf(oldpeak.universe, [0, 0, 2])
    oldpeak['Medium'] = fuzz.trimf(oldpeak.universe, [1, 2, 3])
    oldpeak['High'] = fuzz.trimf(oldpeak.universe, [2, 4, 6])
    
    # ميل ST: صاعد، مسطح، هابط
    slope['Up'] = fuzz.trimf(slope.universe, [0, 0, 1])
    slope['Flat'] = fuzz.trimf(slope.universe, [0, 1, 2])
    slope['Down'] = fuzz.trimf(slope.universe, [1, 2, 2])
    
    # الخطورة: منخفضة، متوسطة، عالية
    risk.automf(3, names=['Low', 'Medium', 'High'])

def define_fuzzy_rules():
    return [
        ctrl.Rule(chol['High'] & trestbps['High'], risk['High']),
        ctrl.Rule(cp['Asymptomatic'] & thalach['Low'], risk['High']),
        ctrl.Rule(age['Young'] & thalach['High'], risk['Low']),
        ctrl.Rule(oldpeak['High'] & slope['Down'], risk['High']),  # تم التصحيح هنا
        ctrl.Rule(sex['Male'] & age['Old'], risk['Medium']),
        ctrl.Rule(chol['Low'] & trestbps['Low'], risk['Low']),
    ]

# تهيئة النظام الضبابي
define_membership_functions()
rules = define_fuzzy_rules()
risk_ctrl = ctrl.ControlSystem(rules)
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

def compute_risk(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    risk_sim.input['age'] = age
    risk_sim.input['sex'] = sex
    risk_sim.input['cp'] = cp
    risk_sim.input['trestbps'] = trestbps
    risk_sim.input['chol'] = chol
    risk_sim.input['thalach'] = thalach
    risk_sim.input['oldpeak'] = oldpeak
    risk_sim.input['slope'] = slope  # إضافة إدخال slope
    risk_sim.compute()
    crisp_value = risk_sim.output['risk']
    return defuzzify_output(crisp_value)

def defuzzify_output(crisp_value):
    if crisp_value <= 3.5:
        return 'Low'
    elif 3.5 < crisp_value <= 6.5:
        return 'Medium'
    else:
        return 'High'