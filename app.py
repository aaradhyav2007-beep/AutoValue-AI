import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from ai_analysis import analyze_car

st.set_page_config(
    page_title="AutoValue AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F5F7FA;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Buttons */
.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 12px;
    border: none;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #1E40AF;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
}

/* Inputs */
.stNumberInput {
    background: white;
}

</style>
""", unsafe_allow_html=True)

st.title("AutoValue AI")

st.markdown("""
### AI-Powered Used Car Valuation Platform

Estimate a vehicle's fair market value using Machine Learning and receive
AI-powered buying recommendations in seconds.
""")

st.sidebar.title("About")

st.sidebar.info(
"""
This project predicts the market value of used cars using Polynomial Regression.
Model performance.
MAE: $2,734 
R2 Score: 91.64%
RMSE: $3,242
"""
)
model = joblib.load("models/polynomial_model.pkl")

transformer = joblib.load("models/polynomial_features.pkl")

left, right = st.columns([1,1])

with left:
    st.markdown("""
    <div style="
    height:430px;
    overflow:hidden;
    border-radius:20px;
    ">
    <img src="https://www.gethow.org/wp-content/uploads/2019/06/resale-car-1000x487.jpg"
    style="
    width:100%;
    height:100%;
    object-fit:cover;
    ">
    </div>
    """, unsafe_allow_html=True)
with right:

    st.subheader(" Vehicle Details")

    age = st.number_input(
        "Age (Years)",
        min_value=0,
        value=5
    )

    mileage = st.number_input(
        "Mileage (KM)",
        min_value=0,
        value=60000
    )

    original_price = st.number_input(
        "Original Price (USD)",
        min_value=0,
        value=30000
    )

    seller_price = st.number_input(
        "Seller Asking Price (Optional)",
        min_value=0,
        value=0
    )

    analyze = st.button(" Analyze Vehicle")

#prediction
if analyze:

    new_car = pd.DataFrame({
    "Age_Years":[age],
    "Mileage_KM":[mileage],
    "Original_Price_USD":[original_price]
})

    new_car_poly = transformer.transform(new_car)

    estimated_price = model.predict(new_car_poly)[0]
    difference = estimated_price - seller_price

    st.subheader("📊 Prediction")

    st.metric(
        label="Estimated Market Price (USD)",
        value=f"${estimated_price:,.2f}"
    )
    #Deal analysis
    st.subheader("💰 Deal Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Seller Asking Price (USD)",
            value=f"${seller_price:,.2f}"
        )
    with col2:
        if estimated_price > seller_price:
            st.metric(
                label="Potential Savings",
                value=f"${difference:,.2f}"
            )
        else:
            st.metric(
                label="Extra Cost",
                value=f"${difference:,.2f}"
        )

    if seller_price > 0:

        if difference > 1000:
            status = "🟢 Great Deal"
            st.success("Great Deal! The asking price is below the estimated market value.")

        elif -1000 <= difference <= 1000:
            status = "🟡 Fair Price"
            st.warning("Fair Price. The asking price is close to the estimated market value.")

        else:
            status = "🔴 Overpriced"
            st.error("Overpriced. Consider negotiating.")


#Depreciation summary
    st.subheader("📉 Depreciation Summary")
    value_lost = original_price - estimated_price
    depreciation = (value_lost / original_price) * 100
    st.write(f"Original Price: ${original_price:,.2f}")
    st.write(f"Estimated Current Price: ${estimated_price:,.2f}")
    st.write(f"Value Lost: ${value_lost:,.2f}")
    st.write(f"Depreciation: {depreciation:.2f}%")

    ai_response = analyze_car(
        age=age,
        mileage=mileage,
        original_price=original_price,
        predicted_price=estimated_price,
        seller_price=seller_price,
        depreciation=depreciation
    )

    st.divider()
    st.subheader("🤖 AI Vehicle Analysis")
    st.write(ai_response)