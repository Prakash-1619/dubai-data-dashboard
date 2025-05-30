import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Real Estate Dashboard & Target Distribution", layout="wide")
st.title("🏙️ Dubai Real Estate Dashboard & Target Distribution")

# --- Load CSV directly from GitHub ---
github_file_url = "new_tdf.csv"

try:
    df = pd.read_csv(github_file_url)
    st.success("✅ Data loaded successfully from GitHub!")
except Exception as e:
    st.error(f"❌ Failed to load data from GitHub: {e}")
    st.stop()

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

st.subheader("🧾 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.sidebar.header("⚙️ Configuration")
target_column = st.sidebar.text_input("🎯 Enter the target column", value="meter_sale_price")

# --- IQR Filtering Function ---
def get_iqr_bounds(df, col):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

# --- Plotting Function ---
def plot_target_distribution_by_object_columns_streamlit(dfs, target, df_names):
    object_cols = [
        'trans_group_en', 'procedure_name_en', 'property_type_en', 'property_sub_type_en',
        'property_usage_en', 'reg_type_en', 'nearest_landmark_en', 'nearest_metro_en',
        'nearest_mall_en', 'rooms_en'
    ]

    for i, df in enumerate(dfs):
        df_name = df_names[i]
        st.header(f"📊 Target Distribution Analysis for: {df_name}")

        if target not in df.columns:
            st.warning(f"Target column '{target}' not found in {df_name}. Skipping this dataset.")
            continue

        fig = px.box(df, y=target, title=f'Overall Boxplot of {target} ({df_name})')
        fig.update_layout(yaxis_title="Meter Sale Price (AED)", yaxis_tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

    for col in object_cols:
        st.subheader(f"📌 Box & Line Plots by: {col}")
        col1, col2 = st.columns(2)

        for i, df in enumerate(dfs):
            df_name = df_names[i]
            if col not in df.columns:
                continue

            with (col1 if i == 0 else col2):
                st.markdown(f"**{df_name}**")

                fig_box = px.box(df, x=col, y=target, title=f'Box Plot by {col} ({df_name})')
                fig_box.update_layout(yaxis_title="Meter Sale Price (AED)", yaxis_tickformat=",")
                st.plotly_chart(fig_box, use_container_width=True)

                year_col = 'instance_year' if 'instance_year' in df.columns else 'instance_Year'
                if year_col in df.columns:
                    grouped_data = df.groupby([year_col, col])[target].mean().reset_index()
                    fig_line = px.line(
                        grouped_data, x=year_col, y=target, color=col,
                        title=f'Line Plot by {year_col} and {col} ({df_name})'
                    )
                    fig_line.update_layout(
                        xaxis_title="Year", yaxis_title="Meter Sale Price (AED)",
                        yaxis_tickformat=",", legend_title=col
                    )
                    fig_line.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    fig_line.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("---")

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Dashboard", "🎯 Target Distribution"])

with tab1:
    st.header("📈 Dashboard Analysis")

    drop_cols = st.multiselect("Select columns to drop", df.columns)
    df_dash = df.drop(columns=drop_cols) if drop_cols else df.copy()

    st.subheader("🧾 Data Summary")
    st.dataframe(pd.DataFrame({
        "Column": df_dash.columns,
        "Data Type": df_dash.dtypes.astype(str),
        "Null Count": df_dash.isnull().sum(),
        "Null %": df_dash.isnull().mean().mul(100).round(2),
        "Unique Values": df_dash.nunique()
    }))

    # --- Univariate Analysis ---
    st.subheader("🔍 Univariate Analysis")
    all_cols = df_dash.columns.tolist()

    if all_cols:
        uni_col = st.selectbox("Select column", all_cols)
        plot_type = st.radio("Plot type", ["Box", "Histogram", "Line", "Bar (Freq)"], horizontal=True)

        try:
            if plot_type in ["Box", "Histogram", "Line"]:
                if pd.api.types.is_numeric_dtype(df_dash[uni_col]):
                    if plot_type == "Box":
                        fig = px.box(df_dash, y=uni_col, title=f"Box Plot of {uni_col}")
                    elif plot_type == "Histogram":
                        fig = px.histogram(df_dash, x=uni_col, title=f"Histogram of {uni_col}")
                    elif plot_type == "Line":
                        fig = px.line(df_dash, y=uni_col, title=f"Line Plot of {uni_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"'{plot_type}' plot is only supported for numeric columns.")
            elif plot_type == "Bar (Freq)":
                freq_df = df_dash[uni_col].value_counts().reset_index()
                freq_df.columns = [uni_col, "Count"]
                fig = px.bar(freq_df, x=uni_col, y="Count", title=f"Frequency of {uni_col}")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Error generating plot for '{uni_col}': {e}")
    else:
        st.warning("No columns available for univariate analysis.")

    if {'instance_year', 'meter_sale_price'}.issubset(df_dash.columns):
        st.subheader("📉 Meter Sale Price Over Years")
        grouped = df_dash.groupby("instance_year")["meter_sale_price"].mean().reset_index()
        counts = df_dash["instance_year"].value_counts().reset_index()
        counts.columns = ["instance_year", "Record Count"]

        fig = px.line(grouped, x="instance_year", y="meter_sale_price", title="Meter Sale Price Trend")
        fig.add_trace(go.Bar(x=counts["instance_year"], y=counts["Record Count"], name="Record Count", yaxis="y2"))

        fig.update_layout(
            yaxis=dict(title="Meter Sale Price"),
            yaxis2=dict(title="Record Count", overlaying="y", side="right"),
            legend=dict(x=0, y=1),
            bargap=0.2
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗺️ Interactive Area-Wise Bubble Map")
    if all(x in df_dash.columns for x in ["area_lat", "area_lon", "meter_sale_price"]):
        df_dash["area_lat"] = pd.to_numeric(df_dash["area_lat"], errors='coerce')
        df_dash["area_lon"] = pd.to_numeric(df_dash["area_lon"], errors='coerce')
        df_dash["meter_sale_price"] = pd.to_numeric(df_dash["meter_sale_price"], errors='coerce')
        df_map = df_dash.dropna(subset=["area_lat", "area_lon", "meter_sale_price"])

        if "transaction_date" in df_map.columns:
            df_map["transaction_date"] = pd.to_datetime(df_map["transaction_date"], errors='coerce')
            min_date, max_date = df_map["transaction_date"].min(), df_map["transaction_date"].max()
            start, end = st.sidebar.date_input("Select Date Range", [min_date, max_date])
            df_map = df_map[(df_map["transaction_date"] >= start) & (df_map["transaction_date"] <= end)]

        if "area_name_en" in df_map.columns:
            areas = sorted(df_map["area_name_en"].dropna().unique())
            selected_areas = st.sidebar.multiselect("Select Areas", areas, default=areas)
            df_map = df_map[df_map["area_name_en"].isin(selected_areas)]

        grouped = df_map.groupby(["area_name_en", "area_lat", "area_lon"])["meter_sale_price"].agg(["count", "mean"]).reset_index()
        grouped.columns = ["Area", "Lat", "Lon", "Record Count", "Avg Meter Price"]

        min_p, max_p = grouped["Avg Meter Price"].min(), grouped["Avg Meter Price"].max()
        price_filter = st.sidebar.slider("Avg. Meter Price Range", float(min_p), float(max_p), (float(min_p), float(max_p)))
        grouped = grouped[(grouped["Avg Meter Price"] >= price_filter[0]) & (grouped["Avg Meter Price"] <= price_filter[1])]

        fig = px.scatter_map(
            grouped, lat="Lat", lon="Lon", size="Record Count", color="Avg Meter Price",
            hover_name="Area", size_max=50, zoom=10, map_style="open-street-map",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📥 Download Filtered Map Data")
        st.download_button(
            label="⬇️ Download CSV",
            data=grouped.to_csv(index=False),
            file_name="filtered_map_data.csv",
            mime="text/csv"
        )
    else:
        st.info("Map requires columns: area_lat, area_lon, meter_sale_price")

with tab2:
    st.header("🎯 Comparative Target Distribution Dashboard")

    try:
        mlower, mupper = get_iqr_bounds(df, 'meter_sale_price')
        plower, pupper = get_iqr_bounds(df, 'procedure_area')

        otdf = df[(df['meter_sale_price'] >= mlower) & (df['meter_sale_price'] <= mupper)]
        odf = otdf[(otdf['procedure_area'] >= plower) & (otdf['procedure_area'] <= pupper)]

        dfs = [df, odf]
        df_names = ['Raw Data', 'Data after Cleaning Outliers']

        if st.button("📊 Generate Target Distribution Plots"):
            st.success(f"Generating plots for target column: **{target_column}**")
            plot_target_distribution_by_object_columns_streamlit(dfs, target_column, df_names)

    except Exception as e:
        st.error(f"❌ Error during IQR filtering or plotting: {e}")
