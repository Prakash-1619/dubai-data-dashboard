import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# Load and clean data dynamically
@st.cache_data
def load_data():
    original = pd.read_csv("original_data.csv")

    # Basic cleaning example — modify as needed
    cleaned = original.copy()
    cleaned.dropna(inplace=True)
    if "meter_sale_price" in cleaned.columns:
        cleaned = cleaned[cleaned["meter_sale_price"] > 0]

    return original, cleaned

original_df, cleaned_df = load_data()

# Sidebar selection
section = st.sidebar.radio("Choose Section", ["Data Preview", "Bubble Map", "Target Distribution"])

if section == "Data Preview":
    st.title("Preview Datasets")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Dataset")
        st.dataframe(original_df.head())
        st.write(original_df.describe())
    with col2:
        st.subheader("Cleaned Dataset")
        st.dataframe(cleaned_df.head())
        st.write(cleaned_df.describe())

elif section == "Bubble Map":
    st.title("Bubble Map Visualization")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Dataset")
        fig1 = px.scatter_mapbox(
            original_df,
            lat="Latitude",
            lon="Longitude",
            size="meter_sale_price",
            color="Location",
            hover_name="Location",
            mapbox_style="carto-positron",
            zoom=10,
            title="Original Data"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Cleaned Dataset")
        fig2 = px.scatter_mapbox(
            cleaned_df,
            lat="Latitude",
            lon="Longitude",
            size="meter_sale_price",
            color="Location",
            hover_name="Location",
            mapbox_style="carto-positron",
            zoom=10,
            title="Cleaned Data"
        )
        st.plotly_chart(fig2, use_container_width=True)

elif section == "Target Distribution":
    st.title("Target Distribution Comparison")
    categorical_cols = [col for col in original_df.columns if original_df[col].dtype == "object"]
    selected_col = st.selectbox("Select Categorical Column", categorical_cols)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.boxplot(data=original_df, x=selected_col, y="meter_sale_price", ax=axes[0, 0])
    axes[0, 0].set_title("Original Boxplot")
    axes[0, 0].tick_params(axis='x', rotation=45)

    sns.lineplot(data=original_df.groupby(selected_col)["meter_sale_price"].mean().reset_index(),
                 x=selected_col, y="meter_sale_price", ax=axes[0, 1], marker="o")
    axes[0, 1].set_title("Original Line Plot")
    axes[0, 1].tick_params(axis='x', rotation=45)

    sns.boxplot(data=cleaned_df, x=selected_col, y="meter_sale_price", ax=axes[1, 0])
    axes[1, 0].set_title("Cleaned Boxplot")
    axes[1, 0].tick_params(axis='x', rotation=45)

    sns.lineplot(data=cleaned_df.groupby(selected_col)["meter_sale_price"].mean().reset_index(),
                 x=selected_col, y="meter_sale_price", ax=axes[1, 1], marker="o")
    axes[1, 1].set_title("Cleaned Line Plot")
    axes[1, 1].tick_params(axis='x', rotation=45)

    st.pyplot(fig)
