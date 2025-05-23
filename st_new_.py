# -*- coding: utf-8 -*-
"""Streamlit App: Compare Original and Cleaned Data"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# --- Page Setup ---
st.set_page_config(page_title="Compare Original vs Cleaned Data", layout="wide")

# --- Load Data ---
df = pd.read_csv("new_tdf.csv")

# --- Remove Outliers Function ---
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

datasets = {'Original DF': df, 'Cleaned ODF': df_clean}

# --- Data Preview and Summary for df only ---
st.title("🔍 Data Preview and Summary")

st.subheader("📄 Original DF Preview (up to 10,000 rows)")
st.dataframe(df.head(10000))  # Show up to 10,000 rows for preview

st.subheader("📋 Data Summary for Original DF")
summary = pd.DataFrame({
    "Column": df.columns,
    "Data Type": [df[col].dtype for col in df.columns],
    "Null Count": df.isnull().sum().values,
    "Null %": (df.isnull().mean().values * 100).round(2),
    "Unique Values": df.nunique().values
})
st.dataframe(summary)

# --- Bubble Map and Target Distribution Tabs ---
tab1, tab2 = st.tabs(["📍 Bubble Map Comparison", "📈 Target Distribution Comparison"])

def prepare_bubble_data(data):
    data = data.copy()
    data['area_lat'] = pd.to_numeric(data['area_lat'], errors='coerce')
    data['area_lon'] = pd.to_numeric(data['area_lon'], errors='coerce')
    data.dropna(subset=['area_lat', 'area_lon', 'meter_sale_price'], inplace=True)
    grouped = data.groupby(['area_name_en', 'area_lat', 'area_lon'])['meter_sale_price'] \
        .agg(['count', 'mean']).reset_index()
    grouped.rename(columns={'count': 'Record Count', 'mean': 'Average Meter Sale Price'}, inplace=True)
    return grouped

with tab1:
    col3, col4 = st.columns(2)
    for col, (name, data) in zip([col3, col4], datasets.items()):
        col.subheader(f"{name}")
        bubble_data = prepare_bubble_data(data)
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
        col.plotly_chart(fig, use_container_width=True)

with tab2:
    target_column = 'meter_sale_price'  # fixed target column

    col5, col6 = st.columns(2)
    for col, (name, data) in zip([col5, col6], datasets.items()):
        col.subheader(f"📊 {name} - Box Plot")
        if 'trans_group_en' in data.columns:
            fig_box = px.box(
                data, x='trans_group_en', y=target_column,
                title=f'Box Plot of {target_column} by Transaction Group'
            )
            col.plotly_chart(fig_box, use_container_width=True)
        else:
            col.warning("'trans_group_en' column is missing")

    col7, col8 = st.columns(2)
    for col, (name, data) in zip([col7, col8], datasets.items()):
        col.subheader(f"📈 {name} - Line Plot")
        year_col = 'instance_year' if 'instance_year' in data.columns else 'instance_Year'
        if year_col in data.columns and 'trans_group_en' in data.columns:
            grouped = data.groupby([year_col, 'trans_group_en'])[target_column].mean().reset_index()
            fig_line = px.line(
                grouped, x=year_col, y=target_column, color='trans_group_en',
                title=f'{target_column} over Time by Transaction Group'
            )
            col.plotly_chart(fig_line, use_container_width=True)
        else:
            col.warning("Required columns missing")
