import joblib
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from keras._tf_keras.keras.models import load_model

# Load the trained model
model = tf.keras.models.load_model('loan_model.h5')
preprocessing = joblib.load("preprocessing_pipeline.pkl")

# Define a function to preprocess user input
# def preprocess_input(data):
#     """
#     Preprocess input data to match the model's expected input format.
#     """
#     processed_data = np.array([[
#         data['no_of_dependents'],
#         data['education'],
#         data['self_employed'],
#         data['income_annum'],
#         data['loan_amount'],
#         data['loan_term'],
#         data['cibil_score'],
#         data['residential_assets_value'],
#         data['commercial_assets_value'],
#         data['luxury_assets_value'],
#         data['bank_asset_value']
#     ]])  # Ensure it's shaped as (1, n_features)
#     return processed_data

# Streamlit app
st.title('Loan Approval Prediction System')

# User inputs
no_of_dependents = st.number_input('Number of Dependents', min_value=0, step=1)
education = st.selectbox('Education Level', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self-Employed', ['No', 'Yes'])
income_annum = st.number_input('Annual Income (in INR)', min_value=0, step=10000)
loan_amount = st.number_input('Loan Amount (in INR)', min_value=0, step=1000)
loan_term = st.number_input('Loan Term (in months)', min_value=0, step=1)
cibil_score = st.slider('CIBIL Score', min_value=300, max_value=900, step=1)
residential_assets_value = st.number_input('Residential Assets Value (in INR)', min_value=0, step=10000)
commercial_assets_value = st.number_input('Commercial Assets Value (in INR)', min_value=0, step=10000)
luxury_assets_value = st.number_input('Luxury Assets Value (in INR)', min_value=0, step=10000)
bank_asset_value = st.number_input('Bank Asset Value (in INR)', min_value=0, step=10000)

# Map user inputs to a dictionary
user_data = {
    'no_of_dependents': no_of_dependents,
    'education': education, # Adjust this encoding based on actual training data encoding
    'self_employed':self_employed,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}

# Preprocess the input data
input_data = pd.DataFrame([user_data])
input_data = preprocessing.transform(input_data)

# Predict loan status
if st.button('Predict Loan Status'):
    try:
        prediction = model.predict(input_data)
        loan_status = 'Approved' if prediction[0][0] > 0.5 else 'Rejected'
        st.write(f'The loan request is **{loan_status}**.')
    except Exception as e:
        st.write(f"Error in prediction: {str(e)}")
