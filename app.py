import numpy as np
import pickle
import streamlit as st
import pandas as pd

# Load the model
with open('model_pickle_2', 'rb') as f:
    model_2 = pickle.load(f)

def wine_prediction(input_data):
    input_df = pd.DataFrame([{
        'fixed acidity': input_data[0],
        'volatile acidity': input_data[1],
        'citric acid': input_data[2],
        'residual sugar': input_data[3],
        'chlorides': input_data[4],
        'free sulfur dioxide': input_data[5],
        'total sulfur dioxide': input_data[6],
        'density': input_data[7],
        'pH': input_data[8],
        'sulphates': input_data[9],
        'alcohol': input_data[10]
    }])

    predicted_quality = model_2.predict(input_df)[0]

    if predicted_quality == 1:
        return "Predicted Wine Quality: Good 🍷"
    else:
        return "Predicted Wine Quality: Bad ❌"


def main():
    st.title("Wine Quality Prediction Web App")

    fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0)
    volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0)
    citric_acid = st.number_input('Citric Acid', min_value=0.0)
    residual_sugar = st.number_input('Residual Sugar', min_value=0.0)
    chlorides = st.number_input('Chlorides', min_value=0.0)
    free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0.0)
    total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0.0)
    density = st.number_input('Density', min_value=0.0)
    pH = st.number_input('pH', min_value=0.0)
    sulphates = st.number_input('Sulphates', min_value=0.0)
    alcohol = st.number_input('Alcohol', min_value=0.0)

    if st.button("Predict Quality"):
        result = wine_prediction([
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            density, pH, sulphates, alcohol
        ])
        st.success(result)

if __name__ == '__main__':
    main()

