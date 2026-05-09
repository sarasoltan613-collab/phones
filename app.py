import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Mobile Phones Dashboard",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Best Selling Mobile Phones 2020.csv")

# =========================
# TRAIN MODEL
# =========================
X = df[["Phone", "Company"]]
y = df["Sold(million)"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         ["Phone", "Company"])
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

model.fit(X, y)

# =========================
# SIDEBAR NAVIGATION
# =========================
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Insights", "Prediction"]
)

# ==================================================
# HOME PAGE
# ==================================================
if page == "Home":

    st.title("📱 Best Selling Mobile Phones 2020")

    st.markdown("""
    ## Welcome To The Dashboard

    This application contains:

    ### 1️⃣ Home
    General information about the project.

    ### 2️⃣ Insights
    Dashboard with:
    - KPIs
    - Charts
    - Filters

    ### 3️⃣ Prediction
    Predict phone sales using Machine Learning.
    """)

    st.dataframe(df.head())

# ==================================================
# INSIGHTS PAGE
# ==================================================
elif page == "Insights":

    st.title("📊 Insights Dashboard")

    # -----------------------------
    # FILTER
    # -----------------------------
    st.sidebar.header("Filters")

    company_filter = st.sidebar.multiselect(
        "Select Company",
        options=df["Company"].unique(),
        default=df["Company"].unique()
    )

    filtered_df = df[
        df["Company"].isin(company_filter)
    ]

    # -----------------------------
    # KPIs
    # -----------------------------
    total_phones = filtered_df.shape[0]

    total_sales = filtered_df["Sold(million)"].sum()

    top_company = (
        filtered_df
        .groupby("Company")["Sold(million)"]
        .sum()
        .idxmax()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Phones",
        total_phones
    )

    col2.metric(
        "Total Sales",
        round(total_sales, 2)
    )

    col3.metric(
        "Top Company",
        top_company
    )

    st.markdown("---")

    # -----------------------------
    # BAR CHART
    # -----------------------------
    fig_bar = px.bar(
        filtered_df,
        x="Phone",
        y="Sold(million)",
        color="Company",
        title="Phone Sales"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

    # -----------------------------
    # PIE CHART
    # -----------------------------
    fig_pie = px.pie(
        filtered_df,
        names="Company",
        values="Sold(million)",
        title="Company Market Share"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

# ==================================================
# PREDICTION PAGE
# ==================================================
elif page == "Prediction":

    st.title("🤖 Prediction Page")

    st.subheader("Enter Inputs")

    phone = st.selectbox(
        "Phone",
        df["Phone"].unique()
    )

    company = st.selectbox(
        "Company",
        df["Company"].unique()
    )

    input_df = pd.DataFrame({
        "Phone": [phone],
        "Company": [company]
    })

    if st.button("Predict Sales"):

        prediction = model.predict(input_df)[0]

        st.success(
            f"Predicted Sales: {round(prediction,2)} Million"
        )
