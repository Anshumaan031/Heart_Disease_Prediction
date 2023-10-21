import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    st.title("Heart Disease Prediction App")
    st.write("""
    This app predicts the probability of having a heart disease!
    Enter the required fields and click on the Predict button.
    """)
    
    # Collect user input
    age = st.slider("Age", 20, 80)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Type 1", "Type 2", "Type 3", "Type 4"])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200)
    chol = st.slider("Serum Cholestoral (mg/dl)", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    thalach = st.slider("Maximum Heart Rate Achieved", 70, 200)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 6.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.slider("Number of Major Vessels (0-3) Colored by Flourosopy", 0, 3)
    thal = st.selectbox("Thal", ["Normal", "Fixed Defect", "Reversable Defect"])

    # One-hot encoding
data = {
    "age": [age],
    "sex": [1 if sex == "Male" else 0],
    # ... (similarly handle other fields)
}

df = pd.DataFrame(data)

# Scale the data
scaler = StandardScaler()
df_scaled = scaler.transform(df)


model = joblib.load('model.pkl')

prediction = model.predict(df_scaled)

if st.button("Predict"):
    st.write(f"The predicted result is {'Positive' if prediction[0] == 1 else 'Negative'} for heart disease.")
    
if __name__ == "__main__":
    main()
