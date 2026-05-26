import streamlit as st
import pickle
import numpy as np

# Load model and scalers
model = pickle.load(open("model.pkl", "rb"))
scaler_X = pickle.load(open("scaler_X.pkl", "rb"))
scaler_y = pickle.load(open("scaler_y.pkl", "rb"))

st.title("House Price Prediction using SVM Regression")

st.write("Predict House Price")

# Inputs
area = st.number_input("Area (sq ft)", min_value=500)

bedrooms = st.number_input("Bedrooms", min_value=1)

bathrooms = st.number_input("Bathrooms", min_value=1)

stories = st.number_input("Stories", min_value=1)

parking = st.number_input("Parking Spaces", min_value=0)

if st.button("Predict Price"):

    data = np.array([[area, bedrooms, bathrooms, stories, parking]])

    scaled_data = scaler_X.transform(data)

    prediction = model.predict(scaled_data)

    final_price = scaler_y.inverse_transform(
        prediction.reshape(-1, 1)
    )

    st.success(f"Predicted House Price: ₹ {final_price[0][0]:,.2f}")