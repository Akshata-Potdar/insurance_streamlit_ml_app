import streamlit as st
import pandas as pd
import pickle


with open("model/insurance_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="💰"
)

st.title("💰 Insurance Cost Prediction")

st.write(
    "Enter customer details to predict medical insurance charges."
)


age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    [
        "southwest",
        "southeast",
        "northwest",
        "northeast"
    ]
)

if st.button("Predict Insurance Cost"):
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Insurance Charges: ${prediction[0]:,.2f}"
    )



