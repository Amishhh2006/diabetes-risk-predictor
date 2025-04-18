import streamlit as st
import numpy as np
import joblib
from PIL import Image
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# App title and description
st.title("Diabetes Risk Prediction Tool")
st.write("""
This application predicts whether a person has a high or low risk of diabetes 
based on diagnostic measurements. Enter your information below and click 'Predict'.
""")

# Create a function to load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e.filename}. Please make sure both 'diabetes_model.pkl' and 'scaler.pkl' are in the current directory.")
        return None, None

# Load the model and scaler
model, scaler = load_model_and_scaler()

# Create input form
st.subheader("Patient Information")

# Use columns to create a more compact layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0, help="Number of times pregnant")
    glucose = st.number_input('Glucose (mg/dL)', min_value=0, max_value=300, value=120, help="Plasma glucose concentration")
    blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200, value=70, help="Diastolic blood pressure")
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20, help="Triceps skin fold thickness")

with col2:
    insulin = st.number_input('Insulin (μU/ml)', min_value=0, max_value=1000, value=80, help="2-Hour serum insulin")
    bmi = st.number_input('BMI (kg/m²)', min_value=0.0, max_value=100.0, value=25.0, step=0.1, help="Body mass index")
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, step=0.01, help="Diabetes pedigree function")
    age = st.number_input('Age (years)', min_value=0, max_value=120, value=30, help="Age in years")

# Create a prediction button
predict_button = st.button('Predict', type='primary')

# Make prediction when the button is clicked
if predict_button:
    try:
        if model is not None and scaler is not None:
            # Format input data as numpy array
            input_data = np.array([
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age
            ]).reshape(1, -1)
            
            # Display input data in a table for confirmation
            st.subheader("Confirmation of Input Data")
            input_df = pd.DataFrame({
                'Feature': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                           'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'],
                'Value': input_data[0]
            })
            st.dataframe(input_df, hide_index=True)
            
            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_data_scaled)
            
            # Display result
            st.subheader("Prediction Result")
            
            if prediction[0] == 1:
                st.error("High Risk ⚠️")
                st.write("""
                The model predicts a **high risk** of diabetes. Please consult with 
                a healthcare professional for proper medical advice and further testing.
                """)
            else:
                st.success("Low Risk ✅")
                st.write("""
                The model predicts a **low risk** of diabetes. However, maintaining a 
                healthy lifestyle is always recommended.
                """)
                
            # Add disclaimer
            st.info("""
            **Disclaimer**: This is a predictive model based on machine learning and should not 
            be used as a substitute for professional medical diagnosis. Always consult with a 
            healthcare provider regarding any health concerns.
            """)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check your input values and try again.")

# Add information about the model
with st.expander("About the Model"):
    st.write("""
    This application uses an XGBoost machine learning model trained on the Pima Indians Diabetes Database.
    The model considers eight features that are important indicators for diabetes prediction:
    
    1. **Pregnancies**: Number of times pregnant
    2. **Glucose**: Plasma glucose concentration (glucose tolerance test)
    3. **Blood Pressure**: Diastolic blood pressure (mm Hg)
    4. **Skin Thickness**: Triceps skin fold thickness (mm)
    5. **Insulin**: 2-Hour serum insulin (mu U/ml)
    6. **BMI**: Body mass index (weight in kg/(height in m)²)
    7. **Diabetes Pedigree Function**: A function that scores likelihood of diabetes based on family history
    8. **Age**: Age in years
    """)
