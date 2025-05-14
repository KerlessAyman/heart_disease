import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from experta import *
from rules import HeartDiseaseExpert
import io

# Load models and feature columns using relative paths
model_path = os.path.join(os.path.dirname(__file__), 'heart_disease_decision_tree_model.joblib')
features_path = os.path.join(os.path.dirname(__file__), 'model_features.joblib')
model = joblib.load(model_path)
model_features = joblib.load(features_path)

st.set_page_config(
    page_title="Heart Disease Detection System",
    layout="wide",
    page_icon="❤️",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background-color: #e9ecef;
        }
        h1, h2, h3 {
            color: #d63384;
        }
        .stButton>button {
            background-color: #d63384;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .stButton>button:hover {
            background-color: #a61d4d;
            color: white;
        }
        .stNumberInput, .stSelectbox, .stSlider {
            margin-bottom: 16px;
        }
        .reportview-container .markdown-text-container {
            font-family: Arial, sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# Helper functions (تبقى كما هي بدون تغيير)
def run_expert_system(patient_data):
    st.write("\n🧠 Running Expert System Evaluation...")
    engine = HeartDiseaseExpert()
    engine.reset()
    for key, value in patient_data.items():
        engine.declare(Fact(**{key: value}))
    engine.run()

def calculate_user_risk_score(user_data):
    score = 0
    risk_mapping = {
        'cholesterol': (user_data['cholesterol'], 250, 30),
        'blood_pressure': (user_data['blood_pressure'], 140, 25),
        'bmi': (user_data['bmi'], 30, 20),
        'glucose': (user_data['glucose'], 126, 15),
        'sleep_hours': (user_data['sleep_hours'], 6, -10),
        'stress_level': (user_data['stress_level'], 'high', 20),
    }

    for factor, params in risk_mapping.items():
        if factor in ['sleep_hours']:
            if params[0] < params[1]:
                score += abs(params[2])
        elif factor == 'stress_level':
            if user_data['stress_level'] == 'high':
                score += params[2]
        else:
            if params[0] >= params[1]:
                score += params[2]
    return min(score, 100)

def run_decision_tree_model(patient_data):
    df_input = pd.DataFrame([patient_data])
    df_input_encoded = pd.get_dummies(df_input)

    for col in model_features:
        if col not in df_input_encoded.columns:
            df_input_encoded[col] = 0

    df_input_encoded = df_input_encoded[model_features]

    prediction = model.predict(df_input_encoded)[0]
    probability = model.predict_proba(df_input_encoded)[0][1] * 100

    user_risk_score = calculate_user_risk_score(patient_data)
    combined_confidence = (probability * 0.7) + (user_risk_score * 0.3)

    if prediction == 1:
        st.error(f"🔴 **Prediction:** Likely to have heart disease\n📈 **Confidence:** {combined_confidence:.2f}%")
    else:
        st.success(f"🟢 **Prediction:** Unlikely to have heart disease\n📉 **Confidence:** {(100 - combined_confidence):.2f}%")

    return prediction, combined_confidence


def sidebar_input():
    st.sidebar.title("🩺 Patient Health Information")
    st.sidebar.markdown("---")
    
    # تنظيم المدخلات في أقسام
    st.sidebar.subheader("Basic Information")
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, step=1)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, format="%.2f")
    family_history = st.sidebar.selectbox("Family history of heart disease?", ["yes", "no"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Medical Metrics")
    cholesterol = st.sidebar.number_input("Cholesterol Level (mg/dL)", min_value=0)
    blood_pressure = st.sidebar.number_input("Blood Pressure (mmHg)", min_value=0)
    glucose = st.sidebar.number_input("Glucose Level (mg/dL)", min_value=0)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Lifestyle Factors")
    smoking = st.sidebar.selectbox("Do you smoke?", ["yes", "no"])
    alcohol = st.sidebar.selectbox("Do you consume alcohol?", ["yes", "no"])
    diet = st.sidebar.selectbox("Diet type", ["healthy", "unhealthy"])
    exercise = st.sidebar.selectbox("Exercise frequency", ["none", "irregular", "regular"])
    physical_activity = st.sidebar.selectbox("Physical Activity Level", ["low", "moderate", "high"])
    sleep_hours = st.sidebar.slider("Average Sleep Hours", 0, 12, 7)
    stress_level = st.sidebar.selectbox("Stress Level", ["low", "medium", "high"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Medical Details")
    cp = st.sidebar.number_input("Chest pain type (1-4)", min_value=1, max_value=4)
    restecg = st.sidebar.number_input("RestECG (0-2)", min_value=0, max_value=2)
    exang = st.sidebar.number_input("Exercise induced angina (0/1)", min_value=0, max_value=1)
    oldpeak = st.sidebar.number_input("Oldpeak", min_value=0.0, format="%.2f")
    slope = st.sidebar.number_input("Slope (1-3)", min_value=1, max_value=3)
    ca = st.sidebar.number_input("CA (0-4)", min_value=0, max_value=4)
    thal = st.sidebar.number_input("Thal (0-3)", min_value=0, max_value=3)
    fbs = st.sidebar.number_input("Fasting blood sugar >120 (0/1)", min_value=0, max_value=1)

    return {
        "age": age,
        "cholesterol": cholesterol,
        "blood_pressure": blood_pressure,
        "smoking": smoking,
        "exercise": exercise,
        "bmi": bmi,
        "glucose": glucose,
        "family_history": family_history,
        "diet": diet,
        "alcohol": alcohol,
        "cp": cp,
        "restecg": restecg,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "fbs": fbs,
        "physical_activity": physical_activity,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level
    }

# تحسين عرض النصائح الصحية
def show_health_tips(prediction):
    st.subheader("💡 Health Recommendations")
    if prediction == 1:
        with st.expander("Click to see recommendations for high risk"):
            st.write("""
            - **Consult a cardiologist** for professional medical advice.
            - **Dietary Changes**: Focus on fruits, vegetables, whole grains, and lean proteins.
            - **Exercise**: Aim for at least 150 minutes of moderate exercise weekly.
            - **Lifestyle**: Quit smoking and limit alcohol consumption.
            - **Stress Management**: Practice meditation or yoga regularly.
            - **Monitoring**: Regularly check your blood pressure, cholesterol, and glucose levels.
            """)
    else:
        with st.expander("Click to see recommendations for maintaining good health"):
            st.write("""
            - **Maintain Healthy Habits**: Continue balanced diet and regular exercise.
            - **Preventive Care**: Regular health check-ups are important.
            - **Sleep Well**: Maintain 7-9 hours of quality sleep.
            - **Stress Reduction**: Keep stress levels in check.
            - **Stay Hydrated**: Drink plenty of water throughout the day.
            """)


def download_report(data, prediction, probability):
    report = pd.DataFrame.from_dict(data, orient='index', columns=['Values'])
    report.loc['Prediction'] = 'Likely' if prediction == 1 else 'Unlikely'
    report.loc['Confidence'] = f"{probability:.2f}%"

    buffer = io.StringIO()
    report.to_csv(buffer)
    st.download_button(
        "📥 Download Full Prediction Report", 
        data=buffer.getvalue(), 
        file_name="heart_disease_report.csv",
        mime="text/csv"
    )


st.title("❤️ Heart Disease Risk Assessment System")
st.markdown("""
    <div style="background-color:#f8d7da;padding:16px;border-radius:8px;margin-bottom:24px;">
        <h4 style="color:#721c24;margin:0;">Welcome to our Heart Disease Risk Assessment Tool</h4>
        <p style="color:#721c24;margin:8px 0 0 0;">
            Please fill in your health information in the sidebar and click "Predict" to assess your heart disease risk.
        </p>
    </div>
""", unsafe_allow_html=True)


col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### How it works:")
    st.markdown("""
        1. Enter your health details in the sidebar
        2. Click the Predict button
        3. View your risk assessment and recommendations
        4. Download your full report
    """)

with col2:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)

user_data = sidebar_input()

if st.sidebar.button("🔍 Predict Heart Disease Risk"):
    st.subheader("📋 Assessment Results")
    
    # استخدام تبويبات لعرض النتائج
    tab1, tab2, tab3 = st.tabs(["Expert System", "Machine Learning", "Visualizations"])
    
    with tab1:
        st.subheader("Expert System Results:")
        run_expert_system(user_data)
    
    with tab2:
        st.subheader("Decision Tree Model Prediction:")
        prediction, combined_confidence = run_decision_tree_model(user_data)
        show_health_tips(prediction)
    
    with tab3:
        st.subheader("📊 Health Metrics Visualization")
        
        # تحسين الرسوم البيانية
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            st.markdown("### Key Health Indicators")
            sample_data = pd.DataFrame({
                'Metric': ['Cholesterol', 'Blood Pressure', 'BMI', 'Glucose', 'Sleep Hours'],
                'Value': [user_data['cholesterol'], user_data['blood_pressure'], 
                         user_data['bmi'], user_data['glucose'], user_data['sleep_hours']]
            })
            
            fig = plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='Metric', y='Value', data=sample_data, palette='Reds_r')
            plt.xticks(rotation=45)
            plt.title("Your Health Metrics Comparison")
            st.pyplot(fig)
        
        with fig_col2:
            st.markdown("### Risk Factor Contribution")
            pie_data = pd.Series({
                'Cholesterol': user_data['cholesterol'] / 3,
                'Blood Pressure': user_data['blood_pressure'] / 2,
                'BMI': user_data['bmi'] * 2,
                'Glucose': user_data['glucose'] / 5,
                'Lifestyle': (12 - user_data['sleep_hours']) * 3 + 
                           (30 if user_data['stress_level'] == 'high' else 10)
            })
            
            fig2 = plt.figure(figsize=(8, 8))
            colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
            ax2 = pie_data.plot.pie(
                autopct='%1.1f%%', 
                startangle=90, 
                colors=colors,
                wedgeprops={'edgecolor':'white', 'linewidth':1}
            )
            plt.title("Risk Factors Distribution")
            plt.ylabel('')
            st.pyplot(fig2)
    
   
    st.markdown("---")
    st.subheader("📄 Full Report")
    download_report(user_data, prediction, combined_confidence)


st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="text-align:center;padding:8px;">
        <p style="font-size:small;color:#6c757d;">
            Developed for academic purposes<br>
            Ready for cloud deployment<br>
            v2.0.0
        </p>
    </div>
""", unsafe_allow_html=True)