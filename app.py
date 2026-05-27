import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# 1. Load the saved model from the app directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'house_price_model.pkl'
if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}\nPlease place house_price_model.pkl in the same folder as app.py.")
    st.stop()

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# 2. Configure App Header
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")
st.title("🏡 House Price Prediction App")
st.write("Enter the configuration of the house below to estimate its market price in Lakhs (INR).")

st.divider()

# 3. Define Input UI fields matching the ranges in your dataset
st.header("📋 House Specifications")

bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=5, value=3, step=1)

size_sqft = st.number_input("Total Area (in Square Feet)", min_value=400, max_value=3000, value=1500, step=50)

age_years = st.slider("Age of the Property (in Years)", min_value=0, max_value=25, value=10, step=1)

st.divider()

# 4. Predict Button Action
if st.button("💰 Calculate Estimated Price", type="primary"):
    # Format inputs exactly as expected by the trained model: [[Bedrooms, Size_sqft, Age_years]]
    input_data = np.array([[bedrooms, size_sqft, age_years]])
    
    # Generate prediction
    predicted_price = model.predict(input_data)[0]
    
    # Enforce a reasonable floor in case extreme inputs result in negative predictions
    if predicted_price < 0:
        predicted_price = 0.0
        
    # Display Result
    st.success(f"### Estimated Market Price: **₹ {predicted_price:.2f} Lakhs**")
  
