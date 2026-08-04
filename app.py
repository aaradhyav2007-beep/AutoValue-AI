"""
AutoValue AI - Used Car Valuation App

Streamlit front-end that:
1. Collects vehicle details from the user
2. Predicts fair market value with a pre-trained sklearn pipeline
   (car_price_pipeline.pkl: OneHotEncoder -> PolynomialFeatures -> LinearRegression)
3. Compares the estimate against the seller's asking price
4. Summarizes depreciation vs. the original purchase price
5. Asks an LLM (via ai_analysis.analyze_car) for a plain-English recommendation
"""

import joblib
import pandas as pd
import streamlit as st

from ai_analysis import analyze_car

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DATA_PATH = "car_data.csv"
PIPELINE_PATH = "car_price_pipeline.pkl"

# Columns the trained pipeline was fit on, in this exact order.
MODEL_FEATURES = [
    "Brand",
    "Model",
    "Car_Type",
    "Transmission",
    "Fuel_Type",
    "Age_Years",
    "No_of_Owners",
    "KM_Driven",
]

st.set_page_config(
    page_title="AutoValue AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        /*
         * Single source of truth for the light theme.
         * Every color used below is defined here — previously the sheet
         * referenced var(--text-light) / var(--text-dark), which were never
         * declared, so those rules silently did nothing. Everything now
         * resolves to one of the tokens below.
         */
        :root {
            --primary-color: #2563EB;      /* Electric Blue */
            --secondary-color: #3B82F6;    /* Sky Blue */
            --accent-color: #06B6D4;       /* Cyan accent for glow/success */
            --background-light: #F8FAFC;
            --surface-glass: rgba(255, 255, 255, 0.75);
            --surface-solid: #FFFFFF;
            --border-glass: rgba(15, 23, 42, 0.08);
            --text-main: #1E293B;          /* Primary body/heading text (dark, on light bg) */
            --text-muted: #64748B;         /* Secondary/caption text */
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--text-main);
        }

        .stApp {
            background: linear-gradient(135deg, #F8FAFC, #EFF6FF, #F8FAFC);
            background-size: 400% 400%;
            animation: gradientAnimation 15s ease infinite;
        }

        @keyframes gradientAnimation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            max-width: 1200px;
            margin: auto;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--primary-color);
            font-weight: 600;
        }

        p, span, label, li {
            color: var(--text-main);
        }

        .stButton>button {
            background-color: var(--secondary-color);
            color: #FFFFFF;
            border-radius: 15px;
            border: none;
            height: 55px;
            width: 100%;
            font-size: 19px;
            font-weight: 600;
            box-shadow: 0px 5px 15px rgba(37, 99, 235, 0.4);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1E40AF;
            box-shadow: 0px 8px 20px rgba(37, 99, 235, 0.6);
            transform: translateY(-2px);
        }

        [data-testid="stMetric"] {
            background: var(--surface-glass);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border-glass);
            padding: 20px;
            border-radius: 20px;
            box-shadow: 0px 8px 25px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            box-shadow: 0px 12px 30px rgba(0,0,0,0.1);
        }
        [data-testid="stMetricLabel"] { color: var(--text-muted); }
        [data-testid="stMetricValue"] { color: var(--primary-color); }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FFFFFF, #F8FAFC);
            border-right: 1px solid var(--border-glass);
            box-shadow: 5px 0px 15px rgba(0,0,0,0.05);
        }
        section[data-testid="stSidebar"] * {
            color: var(--text-main);
        }

        .stNumberInput, .stSelectbox, .stTextInput {
            background: var(--surface-glass);
            border-radius: 10px;
            color: var(--text-main);
        }

        .stInfo {
            background-color: rgba(37, 99, 235, 0.08);
            border-left: 5px solid var(--primary-color);
            border-radius: 8px;
            padding: 15px;
        }

        .stSuccess {
            background-color: rgba(34, 197, 94, 0.1);
            border-left: 5px solid #22C55E;
            border-radius: 8px;
            padding: 15px;
        }

        .stWarning {
            background-color: rgba(250, 204, 21, 0.12);
            border-left: 5px solid #EAB308;
            border-radius: 8px;
            padding: 15px;
        }

        .stError {
            background-color: rgba(239, 68, 68, 0.1);
            border-left: 5px solid #EF4444;
            border-radius: 8px;
            padding: 15px;
        }

        .stSpinner > div {
            border-top-color: var(--primary-color);
            border-left-color: var(--primary-color);
        }

        /* Hero image */
        .hero-image-container {
            height: 430px;
            overflow: hidden;
            border-radius: 25px;
            box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
            border: 1px solid var(--border-glass);
            background: var(--surface-glass);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .hero-image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 25px;
        }

        /* Pulse effect for header */
        .pulse-text {
            animation: pulse 2s infinite alternate;
        }
        @keyframes pulse {
            0% { text-shadow: 0 0 5px rgba(37, 99, 235, 0.3); }
            100% { text-shadow: 0 0 14px rgba(37, 99, 235, 0.55); }
        }

        /* AI analysis panel */
        .ai-analysis-section {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.06), rgba(6, 182, 212, 0.06));
            backdrop-filter: blur(12px);
            border: 2px solid;
            border-image: linear-gradient(45deg, var(--primary-color), var(--accent-color)) 1;
            border-radius: 25px;
            padding: 30px;
            margin-top: 30px;
            box-shadow: 0px 12px 35px rgba(0,0,0,0.06);
            position: relative;
            overflow: hidden;
        }
        .ai-analysis-section h3 {
            color: var(--primary-color);
            margin-bottom: 20px;
        }
        .ai-analysis-section p {
            color: var(--text-main);
            line-height: 1.8;
            font-size: 1.05em;
            /* Was: white-space: nowrap + overflow: hidden + a 40-step typing
               animation. That combo was built for a short one-liner; the
               real AI response is a full paragraph, so it got clipped and/or
               forced horizontal scrolling. Full text now just wraps normally. */
            white-space: normal;
            overflow-wrap: break-word;
            margin: 0;
        }

        .st-divider, hr {
            border-top: 1px solid var(--border-glass);
            margin-top: 2rem;
            margin-bottom: 2rem;
        }

        /* Card-like main content area */
        .main .block-container {
            background: var(--surface-glass);
            backdrop-filter: blur(15px);
            border: 1px solid var(--border-glass);
            border-radius: 30px;
            padding: 40px;
            box-shadow: 0px 15px 40px rgba(0,0,0,0.05);
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline():
    return joblib.load(PIPELINE_PATH)


@st.cache_data(show_spinner=False)
def load_reference_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

# --------------------------------------------------------------------------
# UI sections
# --------------------------------------------------------------------------

def render_header() -> None:
    st.markdown("<h1 class=\"pulse-text\">AutoValue AI</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        ### AI-Powered Used Car Valuation Platform

        Estimate a vehicle's fair market value using Machine Learning and receive
        AI-powered buying recommendations in seconds.
        """
    )


def render_sidebar() -> None:
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This project predicts the market value of used cars using a
        scikit-learn pipeline (categorical encoding + polynomial regression)
        trained on real listing data. See `eda.ipynb` for the full
        exploratory analysis and evaluation metrics.
        """
    )


def render_hero_image() -> None:
    st.markdown(
        """
        <div class="hero-image-container">
        <img src="https://www.gethow.org/wp-content/uploads/2019/06/resale-car-1000x487.jpg"
             alt="Used Car" loading="lazy">
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_vehicle_form(df: pd.DataFrame) -> dict:
    """Collect vehicle details and return them as a plain dict."""
    st.markdown("### Vehicle Details")

    brands = sorted(df["Brand"].unique())
    brand = st.selectbox("Brand", brands)

    models_for_brand = sorted(df.loc[df["Brand"] == brand, "Model"].unique())
    model_name = st.selectbox("Model", models_for_brand)

    car_type = st.selectbox("Car Type", sorted(df["Car_Type"].unique()))
    transmission = st.selectbox("Transmission", sorted(df["Transmission"].unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(df["Fuel_Type"].unique()))

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (Years)", min_value=0, max_value=40, value=5)
        owners = st.number_input("Number of Previous Owners", min_value=1, max_value=10, value=1)
    with col2:
        mileage = st.number_input("Mileage (KM)", min_value=0, value=60000, step=1000)
        original_price = st.number_input(
            "Original Price (USD)", min_value=0, value=30000, step=500
        )

    seller_price = st.number_input(
        "Seller Asking Price (Optional, USD)", min_value=0, value=0, step=500
    )

    analyze = st.button("Analyze Vehicle")

    return {
        "brand": brand,
        "model_name": model_name,
        "car_type": car_type,
        "transmission": transmission,
        "fuel_type": fuel_type,
        "age": age,
        "owners": owners,
        "mileage": mileage,
        "original_price": original_price,
        "seller_price": seller_price,
        "analyze": analyze,
    }

# --------------------------------------------------------------------------
# Prediction & analysis
# --------------------------------------------------------------------------

def predict_price(pipeline, details: dict) -> float:
    row = pd.DataFrame(
        [{
            "Brand": details["brand"],
            "Model": details["model_name"],
            "Car_Type": details["car_type"],
            "Transmission": details["transmission"],
            "Fuel_Type": details["fuel_type"],
            "Age_Years": details["age"],
            "No_of_Owners": details["owners"],
            "KM_Driven": details["mileage"],
        }]
    )[MODEL_FEATURES]

    return float(pipeline.predict(row)[0])

def render_prediction(estimated_price: float) -> None:
    st.markdown("### 📊 Prediction")
    # Was: a hand-rolled div mimicking Streamlit's stMetric with inline
    # styles. That fights with the [data-testid="stMetric"] CSS rules meant
    # for the real widget and bypasses Streamlit's own theming. Use the
    # actual widget instead.
    st.metric(label="Estimated Market Price (USD)", value=f"${estimated_price:,.2f}")

def render_deal_analysis(estimated_price: float, seller_price: float) -> None:
    if seller_price <= 0:
        return  # No asking price entered - nothing to compare against.

    st.markdown("### 💰 Deal Analysis")
    difference = estimated_price - seller_price

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Seller Asking Price (USD)", value=f"${seller_price:,.2f}")
    with col2:
        if estimated_price > seller_price:
            st.metric(label="Potential Savings", value=f"${difference:,.2f}")
        else:
            st.metric(label="Extra Cost", value=f"${-difference:,.2f}")

    if difference > 1000:
        st.success("Great Deal! The asking price is below the estimated market value.")
    elif -1000 <= difference <= 1000:
        st.warning("Fair Price. The asking price is close to the estimated market value.")
    else:
        st.error("Overpriced. Consider negotiating.")

def render_depreciation_summary(original_price: float, estimated_price: float) -> float:
    st.markdown("### 📉 Depreciation Summary")
    value_lost = original_price - estimated_price
    depreciation = (value_lost / original_price * 100) if original_price > 0 else 0.0

    st.write(f"Original Price: ${original_price:,.2f}")
    st.write(f"Estimated Current Price: ${estimated_price:,.2f}")
    st.write(f"Value Lost: ${value_lost:,.2f}")
    st.write(f"Depreciation: {depreciation:.2f}%")

    return depreciation

def render_ai_analysis(details: dict, estimated_price: float, depreciation: float) -> None:
    with st.spinner("Getting AI recommendation..."):
        ai_response = analyze_car(
            model=details["model_name"],
            age=details["age"],
            mileage=details["mileage"],
            original_price=details["original_price"],
            predicted_price=estimated_price,
            seller_price=details["seller_price"],
            depreciation=depreciation,
        )

    st.divider()
    st.markdown("### 🤖 AI Vehicle Analysis")
    st.markdown(
        f"<div class=\"ai-analysis-section\"><p>{ai_response}</p></div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    inject_css()
    render_header()
    render_sidebar()

    pipeline = load_pipeline()
    reference_data = load_reference_data()

    left, right = st.columns([1, 1])
    with left:
        render_hero_image()
    with right:
        details = render_vehicle_form(reference_data)

    if not details["analyze"]:
        return

    estimated_price = predict_price(pipeline, details)

    render_prediction(estimated_price)
    render_deal_analysis(estimated_price, details["seller_price"])
    depreciation = render_depreciation_summary(details["original_price"], estimated_price)

    try:
        render_ai_analysis(details, estimated_price, depreciation)
    except Exception as exc:  # e.g. missing/invalid GEMINI_API_KEY
        st.divider()
        st.markdown("### 🤖 AI Vehicle Analysis")
        st.error(f"AI analysis is currently unavailable: {exc}")

if __name__ == "__main__":
    main()