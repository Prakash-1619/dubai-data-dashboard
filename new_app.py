# -*- coding: utf-8 -*-
# Streamlit App: Data Preview, Summary, and Map Visualization

# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

# --- Streamlit App Title ---
st.title("🔍 Dubai Real Estate Data Preview and Map Visualization")

# --- Load Original Dataset ---
df_path = "new_tdf.csv"
try:
    df = pd.read_csv(df_path)
    st.success(f"Loaded data from {df_path}")
except FileNotFoundError:
    st.error(f"File not found: {df_path}")
    st.stop()

# --- Function to Remove Outliers ---
def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[col] >= lower) & (df[col] <= upper)]

# --- Clean Data ---
df_clean = df.copy()
for col in ['meter_sale_price', 'procedure_area', 'actual_worth']:
    if col in df_clean.columns:
        df_clean = remove_outliers(df_clean, col)

datasets = {'Original DF': df, 'Cleaned DF': df_clean}

# --- Data Preview ---
st.subheader("📄 Original DF Preview (100 Random Rows)")
st.dataframe(df.sample(min(100, len(df))))

# --- Data Summary ---
st.subheader("📋 Data Summary for Original DF")
summary = pd.DataFrame({
    "Column": df.columns,
    "Data Type": [df[col].dtype for col in df.columns],
    "Null Count": df.isnull().sum().values,
    "Null %": (df.isnull().mean().values * 100).round(2),
    "Unique Values": df.nunique().values
})
st.dataframe(summary)

# --- Load Area Plot Stats ---
file_path = "df_area_plot_stats.xlsx"
try:
    df_area_plot_stats = pd.read_excel(file_path)
    st.success(f"Loaded area statistics from {file_path}")
except FileNotFoundError:
    st.error(f"File not found: {file_path}")
    st.stop()

# --- Show Columns for Debug ---
st.write("Columns in area stats dataset:", df_area_plot_stats.columns.tolist())

# --- Plot Map Visualization ---
st.subheader("📍 Dubai Area-wise Average Meter Sale Price and Transaction Count")

if {'area_lat', 'area_lon', 'Transaction Count', 'Average Meter Sale Price', 'area_name_en'}.issubset(df_area_plot_stats.columns):
    fig = px.scatter_mapbox(
        df_area_plot_stats,
        lat='area_lat',
        lon='area_lon',
        size='Transaction Count',
        color='Average Meter Sale Price',
        hover_name='area_name_en',
        hover_data={
            'Transaction Count': True,
            'Average Meter Sale Price': ':.2f',
            'area_lat': False,
            'area_lon': False
        },
        color_continuous_scale='Viridis',
        size_max=30,
        zoom=9,
        title="Dubai Area-wise Average Meter Sale Price and Transaction Count"
    )
    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Required columns not found in the area plot stats file.")


# Box plot

# --- Box Plots: Original vs Cleaned ---
st.subheader("📦 Box Plot Comparison: Original vs Cleaned Data")

cols_to_plot = ['procedure_area', 'meter_sale_price']

for col in cols_to_plot:
    if col in df.columns and col in df_clean.columns:
        st.markdown(f"### 🔍 Box Plot for `{col}`")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Data**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(y=df[col], ax=ax, color='lightblue')
            ax.set_title(f'Original DF: {col}')
            st.pyplot(fig)

        with col2:
            st.markdown("**Cleaned Data**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(y=df_clean[col], ax=ax, color='lightgreen')
            ax.set_title(f'Cleaned DF: {col}')
            st.pyplot(fig)
    else:
        st.warning(f"Column `{col}` not found in one of the datasets.")

