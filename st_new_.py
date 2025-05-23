# -*- coding: utf-8 -*-
"""Cleaned Streamlit App for Multi-File Dashboard"""

# Required Libraries
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Multi-File Data Dashboard", layout="wide")

# --- Load Dataset ---
path = "new_tdf.csv"
df = pd.read_csv(path)

# --- Global Settings ---
seed = 0
target = 'meter_sale_price'

# --- Outlier Removal ---
def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[col] >= lower) & (df[col] <= upper)]

df_clean = remove_outliers(df, 'meter_sale_price')
df_clean = remove_outliers(df_clean, 'procedure_area')
df_clean = remove_outliers(df_clean, 'actual_worth')

# --- Create odf ---
odf = df_clean.copy()

# --- Streamlit UI: Preview ---
st.title("📁 Multi-Source Data Exploration Dashboard")

with st.expander("🔍 Preview Data"):
    st.dataframe(df)
    st.subheader("📋 Data Summary")
    summary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [df[col].dtype for col in df.columns],
        "Null Count": df.isnull().sum().values,
        "Null %": (df.isnull().mean().values * 100).round(2),
        "Unique Values": df.nunique().values
    })
    st.dataframe(summary)

# --- Bubble Map Visualization ---
st.title("📍 Comparative Bubble Maps: DF (Top) vs ODF (Bottom)")

def prepare_bubble_data(data):
    data['area_lat'] = pd.to_numeric(data['area_lat'], errors='coerce')
    data['area_lon'] = pd.to_numeric(data['area_lon'], errors='coerce')
    data.dropna(subset=['area_lat', 'area_lon', 'meter_sale_price'], inplace=True)

    grouped = data.groupby(['area_name_en', 'area_lat', 'area_lon'])['meter_sale_price'] \
        .agg(['count', 'mean']).reset_index()
    grouped.rename(columns={'count': 'Record Count', 'mean': 'Average Meter Sale Price'}, inplace=True)
    return grouped

for label, dataset in {'DF': df, 'ODF': odf}.items():
    st.subheader(f"{label} - Bubble Map")
    bubble_data = prepare_bubble_data(dataset.copy())

    fig = px.scatter_mapbox(
        bubble_data,
        lat='area_lat',
        lon='area_lon',
        hover_name='area_name_en',
        hover_data={'Record Count': True, 'Average Meter Sale Price': True},
        color='Average Meter Sale Price',
        size='Record Count',
        size_max=50,
        color_continuous_scale='Viridis',
        zoom=10,
        mapbox_style='open-street-map'
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Target Distribution Visuals ---
st.title("📈 Comparative Target Distribution by Transaction Group")

dfs = [df, odf]
df_names = ['DF', 'DF after outliers']
target_column = st.text_input("Enter the target column name", value="meter_sale_price")

if st.button("Generate Plots"):
    for data, name in zip(dfs, df_names):
        st.header(f"📊 Distribution for: {name}")

        # --- Box Plot ---
        st.subheader("Box Plot by Transaction Group")
        if 'trans_group_en' in data.columns:
            fig_box = px.box(
                data, x='trans_group_en', y=target_column,
                title=f'Box Plot of {target_column} by Transaction Group ({name})'
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning(f"'trans_group_en' column missing in {name}")

        # --- Line Plot ---
        st.subheader("Line Plot by Year and Transaction Group")
        year_col = 'instance_year' if 'instance_year' in data.columns else 'instance_Year'
        if year_col in data.columns and 'trans_group_en' in data.columns:
            grouped_data = data.groupby([year_col, 'trans_group_en'])[target_column].mean().reset_index()
            fig_line = px.line(
                grouped_data, x=year_col, y=target_column, color='trans_group_en',
                title=f'Year-wise {target_column} by Transaction Group ({name})'
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning(f"Required columns not found in {name}")
