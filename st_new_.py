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
    #transaction group
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
    cat_col = 'trans_group_en'

    if cat_col in df.columns and target_column in df.columns:
        # Data summary grouped by 'trans_group_en'
        summary_df = df.groupby(cat_col)[target_column].describe()

        # Use Streamlit container or column to display summary
        # For example, if inside a column `col`:
        col.markdown("**Data Summary:**")
        col.dataframe(summary_df)
    else:
        st.warning(f"Either '{cat_col}' or '{target_column}' column is missing in the dataframe")



    col5, col6 = st.columns(2)
    for col, (name, data) in zip([col5, col6], datasets.items()):
        col.subheader(f"📊 {name}")
        if cat_col in data.columns:
            fig_box = px.box(
                data, x=cat_col, y=target_column,
                title=f'Box Plot of {target_column} by {cat_col}')
            col.plotly_chart(fig_box, use_container_width=True)
        else:
            col.warning("'trans_group_en' column is missing")

    col7, col8 = st.columns(2)
    for col, (name, data) in zip([col7, col8], datasets.items()):
        #col.subheader(f"📈 {name} - Line Plot")
        year_col = 'instance_year' if 'instance_year' in data.columns else 'instance_Year'
        if year_col in data.columns and cat_col in data.columns:
            grouped = data.groupby([year_col, cat_col])[target_column].mean().reset_index()
            fig_line = px.line(
                grouped, x=year_col, y=target_column, color=cat_col,
                title=f'{target_column} over Time by {cat_col}'
            )
            col.plotly_chart(fig_line, use_container_width=True)
        else:
            col.warning("Required columns missing")

    # property_type_en      
    target_column = 'meter_sale_price'  # fixed target column
    cat_col = 'property_type_en'

    if cat_col in df.columns and target_column in df.columns:
        # Data summary grouped by 'trans_group_en'
        summary_df = df.groupby(cat_col)[target_column].describe()

        # Use Streamlit container or column to display summary
        # For example, if inside a column `col`:
        col.markdown("**Data Summary:**")
        col.dataframe(summary_df)
    else:
        st.warning(f"Either '{cat_col}' or '{target_column}' column is missing in the dataframe")

    col5, col6 = st.columns(2)
    for col, (name, data) in zip([col5, col6], datasets.items()):
        col.subheader(f"📊 {name}")
        if cat_col in data.columns:
            fig_box = px.box(
                data, x=cat_col, y=target_column,
                title=f'Box Plot of {target_column} by {cat_col}'
            )
            col.plotly_chart(fig_box, use_container_width=True)
        else:
            col.warning("'trans_group_en' column is missing")

    col7, col8 = st.columns(2)
    for col, (name, data) in zip([col7, col8], datasets.items()):
        #col.subheader(f"📈 {name} - Line Plot")
        year_col = 'instance_year' if 'instance_year' in data.columns else 'instance_Year'
        if year_col in data.columns and cat_col in data.columns:
            grouped = data.groupby([year_col, cat_col])[target_column].mean().reset_index()
            fig_line = px.line(
                grouped, x=year_col, y=target_column, color=cat_col,
                title=f'{target_column} over Time by {cat_col}'
            )
            col.plotly_chart(fig_line, use_container_width=True)
        else:
            col.warning("Required columns missing")


    # property_sub_type_en   
    target_column = 'meter_sale_price'  # fixed target column
    cat_col = 'property_sub_type_en'

    if cat_col in df.columns and target_column in df.columns:
        # Data summary grouped by 'trans_group_en'
        summary_df = df.groupby(cat_col)[target_column].describe()

        # Use Streamlit container or column to display summary
        # For example, if inside a column `col`:
        col.markdown("**Data Summary:**")
        col.dataframe(summary_df)
    else:
        st.warning(f"Either '{cat_col}' or '{target_column}' column is missing in the dataframe")

    col5, col6 = st.columns(2)
    for col, (name, data) in zip([col5, col6], datasets.items()):
        col.subheader(f"📊 {name}")
        if cat_col in data.columns:
            fig_box = px.box(
                data, x=cat_col, y=target_column,
                title=f'Box Plot of {target_column} by {cat_col}'
            )
            col.plotly_chart(fig_box, use_container_width=True)
        else:
            col.warning("'trans_group_en' column is missing")

    col7, col8 = st.columns(2)
    for col, (name, data) in zip([col7, col8], datasets.items()):
        #col.subheader(f"📈 {name} - Line Plot")
        year_col = 'instance_year' if 'instance_year' in data.columns else 'instance_Year'
        if year_col in data.columns and cat_col in data.columns:
            grouped = data.groupby([year_col, cat_col])[target_column].mean().reset_index()
            fig_line = px.line(
                grouped, x=year_col, y=target_column, color=cat_col,
                title=f'{target_column} over Time by {cat_col}'
            )
            col.plotly_chart(fig_line, use_container_width=True)
        else:
            col.warning("Required columns missing")


    # property_usage_en      
    target_column = 'meter_sale_price'  # fixed target column
    cat_col = 'property_usage_en'

    if cat_col in df.columns and target_column in df.columns:
        # Data summary grouped by 'trans_group_en'
        summary_df = df.groupby(cat_col)[target_column].describe()

        # Use Streamlit container or column to display summary
        # For example, if inside a column `col`:
        col.markdown("**Data Summary:**")
        col.dataframe(summary_df)
    else:
        st.warning(f"Either '{cat_col}' or '{target_column}' column is missing in the dataframe")



    col5, col6 = st.columns(2)
    for col, (name, data) in zip([col5, col6], datasets.items()):
        col.subheader(f"📊 {name}")
        if cat_col in data.columns:
            fig_box = px.box(
                data, x=cat_col, y=target_column,
                title=f'Box Plot of {target_column} by {cat_col}'
            )
            col.plotly_chart(fig_box, use_container_width=True)
        else:
            col.warning("'trans_group_en' column is missing")

    col7, col8 = st.columns(2)
    for col, (name, data) in zip([col7, col8], datasets.items()):
        #col.subheader(f"📈 {name} - Line Plot")
        year_col = 'instance_year' if 'instance_year' in data.columns else 'instance_Year'
        if year_col in data.columns and cat_col in data.columns:
            grouped = data.groupby([year_col, cat_col])[target_column].mean().reset_index()
            fig_line = px.line(
                grouped, x=year_col, y=target_column, color=cat_col,
                title=f'{target_column} over Time by {cat_col}'
            )
            col.plotly_chart(fig_line, use_container_width=True)
        else:
            col.warning("Required columns missing")


    # reg_type_en     
    target_column = 'meter_sale_price'  # fixed target column
    cat_col = 'reg_type_en'

    if cat_col in df.columns and target_column in df.columns:
        # Data summary grouped by 'trans_group_en'
        summary_df = df.groupby(cat_col)[target_column].describe()

        # Use Streamlit container or column to display summary
        # For example, if inside a column `col`:
        col.markdown("**Data Summary:**")
        col.dataframe(summary_df)
    else:
        st.warning(f"Either '{cat_col}' or '{target_column}' column is missing in the dataframe")



    col5, col6 = st.columns(2)
    for col, (name, data) in zip([col5, col6], datasets.items()):
        col.subheader(f"📊 {name}")
        if cat_col in data.columns:
            fig_box = px.box(
                data, x=cat_col, y=target_column,
                title=f'Box Plot of {target_column} by {cat_col}'
            )
            col.plotly_chart(fig_box, use_container_width=True)
        else:
            col.warning("'trans_group_en' column is missing")

    col7, col8 = st.columns(2)
    for col, (name, data) in zip([col7, col8], datasets.items()):
        #col.subheader(f"📈 {name} - Line Plot")
        year_col = 'instance_year' if 'instance_year' in data.columns else 'instance_Year'
        if year_col in data.columns and cat_col in data.columns:
            grouped = data.groupby([year_col, cat_col])[target_column].mean().reset_index()
            fig_line = px.line(
                grouped, x=year_col, y=target_column, color=cat_col,
                title=f'{target_column} over Time by {cat_col}'
            )
            col.plotly_chart(fig_line, use_container_width=True)
        else:
            col.warning("Required columns missing")


    # nearest_landmark_en      
    target_column = 'meter_sale_price'  # fixed target column
    cat_col = 'nearest_landmark_en'

    if cat_col in df.columns and target_column in df.columns:
        # Data summary grouped by 'trans_group_en'
        summary_df = df.groupby(cat_col)[target_column].describe()

        # Use Streamlit container or column to display summary
        # For example, if inside a column `col`:
        col.markdown("**Data Summary:**")
        col.dataframe(summary_df)
    else:
        st.warning(f"Either '{cat_col}' or '{target_column}' column is missing in the dataframe")



    col5, col6 = st.columns(2)
    for col, (name, data) in zip([col5, col6], datasets.items()):
        col.subheader(f"📊 {name}")
        if cat_col in data.columns:
            fig_box = px.box(
                data, x=cat_col, y=target_column,
                title=f'Box Plot of {target_column} by {cat_col}'
            )
            col.plotly_chart(fig_box, use_container_width=True)
        else:
            col.warning("'trans_group_en' column is missing")

    col7, col8 = st.columns(2)
    for col, (name, data) in zip([col7, col8], datasets.items()):
        #col.subheader(f"📈 {name} - Line Plot")
        year_col = 'instance_year' if 'instance_year' in data.columns else 'instance_Year'
        if year_col in data.columns and cat_col in data.columns:
            grouped = data.groupby([year_col, cat_col])[target_column].mean().reset_index()
            fig_line = px.line(
                grouped, x=year_col, y=target_column, color=cat_col,
                title=f'{target_column} over Time by {cat_col}'
            )
            col.plotly_chart(fig_line, use_container_width=True)
        else:
            col.warning("Required columns missing")


    # nearest_metro_en      
    target_column = 'meter_sale_price'  # fixed target column
    cat_col = 'nearest_metro_en'

    if cat_col in df.columns and target_column in df.columns:
        # Data summary grouped by 'trans_group_en'
        summary_df = df.groupby(cat_col)[target_column].describe()

        # Use Streamlit container or column to display summary
        # For example, if inside a column `col`:
        col.markdown("**Data Summary:**")
        col.dataframe(summary_df)
    else:
        st.warning(f"Either '{cat_col}' or '{target_column}' column is missing in the dataframe")



    col5, col6 = st.columns(2)
    for col, (name, data) in zip([col5, col6], datasets.items()):
        col.subheader(f"📊 {name}")
        if cat_col in data.columns:
            fig_box = px.box(
                data, x=cat_col, y=target_column,
                title=f'Box Plot of {target_column} by {cat_col}'
            )
            col.plotly_chart(fig_box, use_container_width=True)
        else:
            col.warning("'trans_group_en' column is missing")

    col7, col8 = st.columns(2)
    for col, (name, data) in zip([col7, col8], datasets.items()):
        #col.subheader(f"📈 {name} - Line Plot")
        year_col = 'instance_year' if 'instance_year' in data.columns else 'instance_Year'
        if year_col in data.columns and cat_col in data.columns:
            grouped = data.groupby([year_col, cat_col])[target_column].mean().reset_index()
            fig_line = px.line(
                grouped, x=year_col, y=target_column, color=cat_col,
                title=f'{target_column} over Time by {cat_col}'
            )
            col.plotly_chart(fig_line, use_container_width=True)
        else:
            col.warning("Required columns missing")


    # nearest_mall_en       
    target_column = 'meter_sale_price'  # fixed target column
    cat_col = 'nearest_mall_en'

    if cat_col in df.columns and target_column in df.columns:
        # Data summary grouped by 'trans_group_en'
        summary_df = df.groupby(cat_col)[target_column].describe()

        # Use Streamlit container or column to display summary
        # For example, if inside a column `col`:
        col.markdown("**Data Summary:**")
        col.dataframe(summary_df)
    else:
        st.warning(f"Either '{cat_col}' or '{target_column}' column is missing in the dataframe")



    col5, col6 = st.columns(2)
    for col, (name, data) in zip([col5, col6], datasets.items()):
        col.subheader(f"📊 {name}")
        if cat_col in data.columns:
            fig_box = px.box(
                data, x=cat_col, y=target_column,
                title=f'Box Plot of {target_column} by {cat_col}'
            )
            col.plotly_chart(fig_box, use_container_width=True)
        else:
            col.warning("'trans_group_en' column is missing")

    col7, col8 = st.columns(2)
    for col, (name, data) in zip([col7, col8], datasets.items()):
        #col.subheader(f"📈 {name} - Line Plot")
        year_col = 'instance_year' if 'instance_year' in data.columns else 'instance_Year'
        if year_col in data.columns and cat_col in data.columns:
            grouped = data.groupby([year_col, cat_col])[target_column].mean().reset_index()
            fig_line = px.line(
                grouped, x=year_col, y=target_column, color=cat_col,
                title=f'{target_column} over Time by {cat_col}'
            )
            col.plotly_chart(fig_line, use_container_width=True)
        else:
            col.warning("Required columns missing")


    # rooms_en      
    target_column = 'meter_sale_price'  # fixed target column
    cat_col = 'rooms_en'

    if cat_col in df.columns and target_column in df.columns:
        # Data summary grouped by 'trans_group_en'
        summary_df = df.groupby(cat_col)[target_column].describe()

        # Use Streamlit container or column to display summary
        # For example, if inside a column `col`:
        col.markdown("**Data Summary:**")
        col.dataframe(summary_df)
    else:
        st.warning(f"Either '{cat_col}' or '{target_column}' column is missing in the dataframe")



    col5, col6 = st.columns(2)
    for col, (name, data) in zip([col5, col6], datasets.items()):
        col.subheader(f"📊 {name}")
        if cat_col in data.columns:
            fig_box = px.box(
                data, x=cat_col, y=target_column,
                title=f'Box Plot of {target_column} by {cat_col}'
            )
            col.plotly_chart(fig_box, use_container_width=True)
        else:
            col.warning("'trans_group_en' column is missing")

    col7, col8 = st.columns(2)
    for col, (name, data) in zip([col7, col8], datasets.items()):
        #col.subheader(f"📈 {name} - Line Plot")
        year_col = 'instance_year' if 'instance_year' in data.columns else 'instance_Year'
        if year_col in data.columns and cat_col in data.columns:
            grouped = data.groupby([year_col, cat_col])[target_column].mean().reset_index()
            fig_line = px.line(
                grouped, x=year_col, y=target_column, color=cat_col,
                title=f'{target_column} over Time by {cat_col}'
            )
            col.plotly_chart(fig_line, use_container_width=True)
        else:
            col.warning("Required columns missing")


    # has_parking      
    target_column = 'meter_sale_price'  # fixed target column
    cat_col = 'has_parking'

    if cat_col in df.columns and target_column in df.columns:
        # Data summary grouped by 'trans_group_en'
        summary_df = df.groupby(cat_col)[target_column].describe()

        # Use Streamlit container or column to display summary
        # For example, if inside a column `col`:
        col.markdown("**Data Summary:**")
        col.dataframe(summary_df)
    else:
        st.warning(f"Either '{cat_col}' or '{target_column}' column is missing in the dataframe")



    col5, col6 = st.columns(2)
    for col, (name, data) in zip([col5, col6], datasets.items()):
        col.subheader(f"📊 {name}")
        if cat_col in data.columns:
            fig_box = px.box(
                data, x=cat_col, y=target_column,
                title=f'Box Plot of {target_column} by {cat_col}'
            )
            col.plotly_chart(fig_box, use_container_width=True)
        else:
            col.warning("'trans_group_en' column is missing")

    col7, col8 = st.columns(2)
    for col, (name, data) in zip([col7, col8], datasets.items()):
        #col.subheader(f"📈 {name} - Line Plot")
        year_col = 'instance_year' if 'instance_year' in data.columns else 'instance_Year'
        if year_col in data.columns and cat_col in data.columns:
            grouped = data.groupby([year_col, cat_col])[target_column].mean().reset_index()
            fig_line = px.line(
                grouped, x=year_col, y=target_column, color=cat_col,
                title=f'{target_column} over Time by {cat_col}'
            )
            col.plotly_chart(fig_line, use_container_width=True)
        else:
            col.warning("Required columns missing")
