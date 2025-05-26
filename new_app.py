import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(layout="wide")
st.title("🔍 Dubai Real Estate Dashboard")

# --- File Paths ---
df_path = "new_tdf.csv"
area_stats_path = "df_area_plot_stats.xlsx"
cat_plot_path = "original_df_description_year.xlsx"

# --- Load Data with Error Handling ---
@st.cache_data
def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.sidebar.error(f"File not found: {file_path}")
        st.stop()

@st.cache_data
def load_excel(file_path):
    try:
        return pd.read_excel(file_path)
    except FileNotFoundError:
        st.sidebar.error(f"File not found: {file_path}")
        st.stop()

# --- Load Main Dataset ---
df = load_csv(df_path)
st.sidebar.success("Main data loaded.")

# --- Remove Outliers ---
def remove_outliers(df, col):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

df_clean = df.copy()
for col in ['meter_sale_price', 'procedure_area', 'actual_worth']:
    if col in df_clean.columns:
        df_clean = remove_outliers(df_clean, col)

# --- Load Area Stats ---
df_area_plot_stats = load_excel(area_stats_path)
st.sidebar.success("Area stats loaded.")

# --- Sidebar Navigation ---
sidebar_option = st.sidebar.radio("Choose View", [
    "Data Preview",
    "Map Visualization",
    "Plots on Categorical Columns"
])

# --- View 1: Data Preview ---
if sidebar_option == "Data Preview":
    tab1, tab2, tab3 = st.tabs(["Preview", "Summary", "Box Plots"])

    with tab1:
        st.subheader("📄 Original DF Preview (100 Random Rows)")
        st.dataframe(df.sample(min(100, len(df))))

    with tab2:
        st.subheader("📋 Data Summary for Original DF")
        summary = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Null Count": df.isnull().sum().values.astype(int),
        "Null %": (df.isnull().mean().values * 100).round(2).astype(float),
        "Unique Values": df.nunique().values.astype(int)
        })


    with tab3:
        st.subheader("📦 Box Plot Comparison: Original vs Cleaned Data")
        for col in ['procedure_area', 'meter_sale_price']:
            if col in df.columns and col in df_clean.columns:
                st.markdown(f"### 🔍 `{col}`")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original**")
                    fig1 = go.Figure(go.Box(y=df[col], name='Original', boxmean='sd', marker_color='royalblue'))
                    fig1.update_layout(yaxis_title=col)
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    st.markdown("**Cleaned**")
                    fig2 = go.Figure(go.Box(y=df_clean[col], name='Cleaned', boxmean='sd', marker_color='seagreen'))
                    fig2.update_layout(yaxis_title=col)
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning(f"Column `{col}` not found in both datasets.")

# --- View 2: Map Visualization ---
elif sidebar_option == "Map Visualization":
    st.subheader("📍 Dubai Area-wise Avg. Meter Sale Price and Transaction Count")
    required_cols = {'area_lat', 'area_lon', 'Transaction Count', 'Average Meter Sale Price', 'area_name_en'}

    if required_cols.issubset(df_area_plot_stats.columns):
        fig = px.scatter_mapbox(
            df_area_plot_stats,
            lat='area_lat',
            lon='area_lon',
            size='Transaction Count',
            color='Average Meter Sale Price',
            hover_name='area_name_en',
            hover_data={'Transaction Count': True, 'Average Meter Sale Price': ':.2f'},
            color_continuous_scale='Viridis',
            size_max=30,
            zoom=9
        )
        fig.update_layout(mapbox_style='open-street-map', margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required columns not found in area stats file.")

# --- View 3: Plots on Categorical Columns ---
elif sidebar_option == "Plots on Categorical Columns":
    st.subheader("📊 Box Plot and Mean Line Plot by Categorical Columns")

    try:
        xls = pd.ExcelFile(cat_plot_path)
        sheet_names = xls.sheet_names
    except FileNotFoundError:
        st.error(f"File not found: {cat_plot_path}")
        st.stop()

    sheet = st.selectbox("Select Sheet to Visualize", sheet_names)
    df_plot = pd.read_excel(xls, sheet_name=sheet)
    st.write(f"### Sheet: {sheet}")
    st.dataframe(df_plot)

    def plot_boxplot(df):
        stats_cols = ['min', '25%', '50%', '75%', 'max']
        if not all(col in df.columns for col in stats_cols):
            return None

        group_col = df.columns[1]
        fig = go.Figure()
        for _, row in df.iterrows():
            fig.add_trace(go.Box(
                y=[row['min'], row['25%'], row['50%'], row['75%'], row['max']],
                name=str(row[group_col]),
                boxpoints=False
            ))
        fig.update_layout(
            title=f'Box Plot by {group_col}',
            yaxis_title='Value',
            xaxis_title=group_col
        )
        return fig

    def plot_mean_line(df):
        if 'instance_year' not in df.columns or 'mean' not in df.columns:
            return None

        legend_col = df.columns[2] if len(df.columns) > 2 else None
        fig = go.Figure()

        if legend_col and legend_col in df.columns:
            for name, group_df in df.groupby(legend_col):
                fig.add_trace(go.Scatter(
                    x=group_df['instance_year'],
                    y=group_df['mean'],
                    mode='lines+markers',
                    name=f'{name} - Mean'
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df['instance_year'],
                y=df['mean'],
                mode='lines+markers',
                name='Mean'
            ))

        fig.update_layout(
            title=f'Mean over Years by {legend_col if legend_col else "N/A"}',
            xaxis_title='Instance Year',
            yaxis_title='Mean',
            hovermode='x unified'
        )
        return fig

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Box Plot")
        box_fig = plot_boxplot(df_plot)
        st.plotly_chart(box_fig, use_container_width=True) if box_fig else st.info("Box plot not available.")

    with col2:
        st.subheader("📈 Mean Line Plot")
        line_fig = plot_mean_line(df_plot)
        st.plotly_chart(line_fig, use_container_width=True) if line_fig else st.info("Mean line plot not available.")
