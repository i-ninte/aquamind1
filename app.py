import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

def get_water_quality_prediction(user_data, model, scaler):
    user_data = user_data.fillna(user_data.median())
    user_data.columns = user_data.columns.str.lower().str.replace('.', '')
    X = user_data
    X = scaler.transform(X)
    prediction = model.predict(X)
    return prediction

# Streamlit app
st.title('Water Quality Prediction')

st.markdown("""
<style>
.main {
    background-color: #f0f2f6;
}
.reportview-container .markdown-text-container {
    font-family: 'Arial', sans-serif;
    color: #4c4c4c;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.header('Input Features')
temperature = st.sidebar.number_input("Temperature", min_value=0.0, step=0.1)
do = st.sidebar.number_input("Dissolved Oxygen (DO)", min_value=0.0, step=0.1)
ph = st.sidebar.number_input("pH Value", min_value=0.0, step=0.1)
conductivity = st.sidebar.number_input("Conductivity", min_value=0.0, step=0.1)
bod = st.sidebar.number_input("Biochemical Oxygen Demand (BOD)", min_value=0.0, step=0.1)

user_data = {
    'temperature': [temperature],
    'do': [do],
    'ph': [ph],
    'conductivity': [conductivity],
    'bod': [bod]
}

input_df = pd.DataFrame(user_data)

if st.sidebar.button('Predict Water Quality'):
    prediction = get_water_quality_prediction(input_df, model, scaler)
    quality = "Potable" if prediction[0] == 1 else "Not Potable"
    st.subheader(f'Predicted Water Quality: {quality}')

    if prediction[0] == 1:
        st.success('The water is potable.')
    else:
        st.error('The water is not potable.')

st.markdown("""
### Water Quality Prediction App
This app predicts the quality of water based on various input features.
Adjust the features in the sidebar and click 'Predict Water Quality' to see the prediction.
""")
