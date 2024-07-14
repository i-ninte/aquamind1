import streamlit as st
import requests
from pydantic import BaseModel

# Define the input form for water quality parameters
class WaterQualityInput(BaseModel):
    temperature: float
    do: float
    ph: float
    conductivity: float
    bod: float

# Function to get predictions from the FastAPI backend
def get_prediction(input_data):
    url = "http://127.0.0.1:8000/predict"  # Update this to your localhost URL
    response = requests.post(url, json=input_data.dict())
    return response.json()

# Streamlit app layout
st.title("AquaMind Water Quality Prediction")

# Collect user inputs
temperature = st.number_input("Enter the temperature value:", min_value=0.0, max_value=100.0, value=25.0)
do = st.number_input("Enter the dissolved oxygen value:", min_value=0.0, max_value=20.0, value=7.0)
ph = st.number_input("Enter the pH value:", min_value=0.0, max_value=14.0, value=7.0)
conductivity = st.number_input("Enter the conductivity value:", min_value=0.0, max_value=1000.0, value=300.0)
bod = st.number_input("Enter the BOD value:", min_value=0.0, max_value=100.0, value=3.0)

# Submit button
if st.button("Predict"):
    input_data = WaterQualityInput(
        temperature=temperature,
        do=do,
        ph=ph,
        conductivity=conductivity,
        bod=bod
    )
    result = get_prediction(input_data)
    
    # Display the prediction result
    st.subheader("Prediction Result")
    st.write(f"Predicted water quality: {result['quality']}")
    st.write(f"Prediction confidence: {result['confidence']:.2f}")
    st.write(f"Class probabilities: {result['probabilities']}")

    # Display the input data for verification
    st.subheader("Input Data")
    st.write(input_data.dict())
