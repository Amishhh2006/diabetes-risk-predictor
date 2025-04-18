DIABETES RISK PREDICTOR (DRP)
=============================

A Streamlit web application that predicts diabetes risk using machine learning (XGBoost) trained on the Pima Indians Diabetes dataset.

REQUIREMENTS
------------
- Python 3.8+
- Streamlit 1.32+
- XGBoost 2.0+
- scikit-learn 1.3+
- pandas 2.1+
- matplotlib 3.8+

INSTALLATION
------------
1. Clone the repository:
   git clone https://github.com/yourusername/diabetes-risk-predictor.git

2. Navigate to project directory:
   cd diabetes-risk-predictor

3. Install dependencies:
   pip install -r requirements.txt

4. (First-time setup) Train the model:
   python train_model.py

USAGE
-----
To launch the web application:
   streamlit run streamlit_app.py

The app will open in your default browser at:
   http://localhost:8501

INPUT PARAMETERS
----------------
- Pregnancies (0-20)
- Glucose level (50-300 mg/dL)
- Blood Pressure (30-140 mmHg)
- Skin Thickness (0-100 mm)
- Insulin (0-300 μU/mL)
- BMI (10.0-50.0)
- Diabetes Pedigree Function (0.0-2.5)
- Age (20-100 years)

OUTPUT
------
- Risk probability percentage (0-100%)
- Risk classification:
  * LOW (0-30%)
  * MODERATE (30-70%)
  * HIGH (70-100%)
- Visual risk meter
- Feature importance chart

FILES
-----
streamlit_app.py    - Main application code
train_model.py      - Model training script
diabetes_model.pkl  - Pretrained XGBoost model
scaler.pkl          - Feature scaling object
requirements.txt    - Python dependencies

DEPLOYMENT
----------
To deploy on Streamlit Sharing:
1. Push to GitHub repository
2. Sign in to share.streamlit.io
3. Connect your repository
4. Deploy!

DISCLAIMER
----------
THIS APPLICATION IS FOR EDUCATIONAL PURPOSES ONLY AND DOES NOT PROVIDE MEDICAL ADVICE. The predictions are based on a machine learning model and should not replace professional medical evaluation. Always consult with a qualified healthcare provider for medical concerns.

SUPPORT
-------
For issues or questions, please contact:
your.email@example.com

LICENSE
-------
MIT License - See LICENSE file for details.
