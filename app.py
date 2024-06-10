import streamlit as st
import pandas as pd
import requests

st.title('AQUAMIND')

st.sidebar.header('Input Features')
temperature = st.sidebar.number_input("Temperature", min_value=0.0, step=0.1)
do = st.sidebar.number_input("Dissolved Oxygen (DO)", min_value=0.0, step=0.1)
ph = st.sidebar.number_input("pH Value", min_value=0.0, step=0.1)
conductivity = st.sidebar.number_input("Conductivity", min_value=0.0, step=0.1)
bod = st.sidebar.number_input("Biochemical Oxygen Demand (BOD)", min_value=0.0, step=0.1)

user_data = {
    'temperature': temperature,
    'do': do,
    'ph': ph,
    'conductivity': conductivity,
    'bod': bod
}

if st.sidebar.button('Predict Water Quality'):
    response = requests.post("http://127.0.0.1:8000/predict", json=user_data)
    if response.status_code == 200:
        prediction = response.json()['quality']
        st.subheader(f'Predicted Water Quality: {prediction}')
        if prediction == "Potable":
            st.success('The water is potable.')
        else:
            st.error('The water is not potable.')
    else:
        st.error(f"Error: {response.status_code}, {response.text}")

st.markdown("""
### AQUAMIND
This app predicts the quality of water based on various input features.
Adjust the features in the sidebar and click 'Predict Water Quality' to see the prediction.
""")
