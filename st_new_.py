# -*- coding: utf-8 -*-
"""Streamlit App: Fully Precomputed Compare Original and Cleaned Data"""

import pandas as pd
import streamlit as st
import plotly.express as px

# --- Page Setup ---
st.set_page_config(page_title="Compare Original vs Cleaned Data", layout="wide")

# --- Remove Outliers Function ---
def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[col] >= lower) & (df[col] <= upper)]

# --- Cached Data Loading and Precomputation ---
@st.cache_data(show_spinner=True)
def load_data(path):
    df = pd.read_csv(path)
    return df

@st.cache_data(show_spinner=True)
def clean_data(df):
    df_clean = df.copy()
    for col in ['meter_sale_price', 'procedure_area', 'actual_worth']:
        if col in df_clean.columns:
            df_clean = remove_outliers(df_clean, col)
    return df_clean

@st.cache_data(show_spinner=True)
def prepare_bubble_data(data):
    data = data.copy()
    data['area_lat'] = pd.to_numeric(data['area_lat'], errors='coerce')
    data['area_lon'] = pd.to_numeric(data['area_lon'], errors='coerce')
    data.dropna(subset=['area_lat', 'area_lon', 'meter_sale_price'], inplace=True)
    grouped = data.groupby(['area_name_en', 'area_lat', 'area_lon'])['meter_sale_price'] \
        .agg(['count', 'mean']).reset_index()
    grouped.rename(columns={'count': 'Record Count', 'mean': 'Average Meter Sale Price'}, inplace=True)
    return grouped

@st.cache_data(show_spinner=True)
def generate_summary(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": [df[col].dtype for col in df.columns],
        "Null Count": df.isnull().sum().values,
        "Null %": (df.isnull().mean().values * 100).round(2),
        "Unique Values": df.nunique().values
    })

@st.cache_data(show_spinner=True)
def generate_bubble_fig(bubble_data, title):
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
        mapbox_style='open-street-map',
        title=title
    )
    return fig

@st.cache_data(show_spinner=True)
def generate_box_plot(data, target_column, title):
    if 'trans_group_en' in data.columns:
        fig_box = px.box(
            data, x='trans_group_en', y=target_column,
            title=title
        )
        return fig_box
    return None

@st.cache_data(show_spinner=True)
def generate_line_plot(data, target_column, title):
    year_col = 'instance_year' if 'instance_year' in data.columns else 'instance_Year'
    if year_col in data.columns and 'trans_group_en' in data.columns:
        grouped = data.groupby([year_col, 'trans_group_en'])[target_column].mean().reset_index()
        fig_line = px.line(
            grouped, x=year_col, y=target_column, color='trans_group_en',
            title=title
        )
        return fig_line
    return None

# --- Load and prepare all datasets and plots ---
df = load_data("new_tdf.csv")
df_clean = clean_data(df)

summary = generate_summary(df)

bubble_data_original = prepare_bubble_data(df)
bubble_data_cleaned = prepare_bubble_data(df_clean)

fig_bubble_original = generate_bubble_fig(bubble_data_original, "Original DF Bubble Map")
fig_bubble_cleaned = generate_bubble_fig(bubble_data_cleaned, "Cleaned ODF Bubble Map")

target_column = "meter_sale_price"

fig_box_orig = generate_box_plot(df, target_column, f"Box Plot of {target_column} by Transaction Group (Original DF)")
fig_line_orig = generate_line_plot(df, target_column, f"{target_column} Over Time by Transaction Group (Original DF)")

fig_box_clean = generate_box_plot(df_clean, target_column, f"Box Plot of {target_column} by Transaction Group (Cleaned ODF)")
fig_line_clean = generate_line_plot(df_clean, target_column, f"{target_column} Over Time by Transaction Group (Cleaned ODF)")

# --- UI ---

st.title("🔍 Data Preview and Summary (Original DF only)")

st.subheader("📄 Original DF Preview (first 1000 rows)")
st.dataframe(df.head(1000))

st.subheader("📋 Original DF Summary")
st.dataframe(summary)

st.title("📍 Bubble Map Comparison")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original DF")
    st.plotly_chart(fig_bubble_original, use_container_width=True)
with col2:
    st.subheader("Cleaned ODF")
    st.plotly_chart(fig_bubble_cleaned, use_container_width=True)

st.title("📈 Target Distribution Comparison")
tab1, tab2 = st.tabs(["Original DF", "Cleaned ODF"])

with tab1:
    st.subheader("Box Plot by Transaction Group (Original DF)")
    if fig_box_orig:
        st.plotly_chart(fig_box_orig, use_container_width=True)
    else:
        st.warning("'trans_group_en' column missing or insufficient data in Original DF")

    st.subheader("Line Plot Over Time by Transaction Group (Original DF)")
    if fig_line_orig:
        st.plotly_chart(fig_line_orig, use_container_width=True)
    else:
        st.warning("Required columns missing or insufficient data in Original DF")

with tab2:
    st.subheader("Box Plot by Transaction Group (Cleaned ODF)")
    if fig_box_clean:
        st.plotly_chart(fig_box_clean, use_container_width=True)
    else:
        st.warning("'trans_group_en' column missing or insufficient data in Cleaned ODF")

    st.subheader("Line Plot Over Time by Transaction Group (Cleaned ODF)")
    if fig_line_clean:
        st.plotly_chart(fig_line_clean, use_container_width=True)
    else:
        st.warning("Required columns missing or insufficient data in Cleaned ODF")
