import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- Page Config ---
st.set_page_config(layout="wide")
st.title("🔍 Dubai Real Estate Dashboard")

# --- File Paths ---
df_path = "target_df.csv"
area_stats_path = "df_area_plot_stats.xlsx"
cat_plot_path = "original_df_description_year.xlsx"
summary = "data_summary.xlsx"
sample = "sample_df.csv"
cat_plot_path_clean = "df_clean_description_data_year.xlsx"

# --- Load Data with Error Handling ---

def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.sidebar.error(f"File not found: {file_path}")
        st.stop()

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
for col in ['meter_sale_price', 'procedure_area']:
    if col in df_clean.columns:
        df_clean = remove_outliers(df_clean, col)

# --- Load Area Stats ---
df_area_plot_stats = load_excel(area_stats_path)
st.sidebar.success("Area stats loaded.")

# --- Sidebar Navigation ---
sidebar_option = st.sidebar.radio("Choose View", [
    "Data Preview",
    "Map Visualization",
    "Plots on Categorical Columns",
    "Model Output"
])

# --- View 1: Data Preview ---
if sidebar_option == "Data Preview":
    tab1, tab2, tab3 = st.tabs(["Preview", "Summary", "Distribution & Box Plots"])

    with tab1:
        sample_df = pd.read_csv(sample)
        st.subheader("📄 Original DF Preview")
        st.dataframe(sample_df)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="No of Columns", value=46)
            st.metric(label="Total Records", value=1424588)
        with col2:
            st.markdown("## Date Column : Instance_data")
            st.metric(label="Start Date", value="1966-01-18")
            st.metric(label="Last date", value="2025-04-03")
        
        st.subheader("📋 Data Summary for Original DF")
        summary_df = pd.read_excel(summary)
        st.dataframe(summary_df)

        try:
            pereto_file = "pereto_analysis_file.xlsx"
            html_pereto_df = "pareto_analysis_plot.html"
            html_pereto_df_clean = "pareto_analysis_plot_after_model_run.html"
            pereto_analyis = pd.ExcelFile(pereto_file)
            pereto_sheet_names = pereto_analyis.sheet_names

        except FileNotFoundError:
            st.error(f"File not found: {pereto_file}")
            st.stop()

        pereto_sheet = st.selectbox("Select Sheet to Show", pereto_sheet_names)
        pereto_df = pd.read_excel(pereto_analyis, sheet_name=pereto_sheet)

        if pereto_sheet == "Original_df":
            st.dataframe(pereto_df)
            st.markdown("## Pereto_analysis_original_df")
            if os.path.exists(html_pereto_df):
                with open(html_pereto_df, "r", encoding="utf-8") as f:
                    dt_html = f.read()
                components.html(dt_html, height=400, scrolling=True)

        elif pereto_sheet == "Data_for_model_run":
            st.dataframe(pereto_df)
            st.markdown("## Pereto_analysis_after_model_run")
            if os.path.exists(html_pereto_df_clean):
                with open(html_pereto_df_clean, "r", encoding="utf-8") as f:
                    dt_html = f.read()
                components.html(dt_html, height=400, scrolling=True)



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



        if 'instance_year' in df.columns and 'meter_sale_price' in df.columns:
            st.markdown("### 📊 Avg Meter Sale Price & Distribution by Instance Year (Original Data)")

            # Group data
            agg_data = df.groupby('instance_year')['meter_sale_price'].agg(['mean', 'count']).reset_index()

            # Create subplot with secondary y-axis
            fig_combo = make_subplots(specs=[[{"secondary_y": True}]])

            # Add line plot for average price
            fig_combo.add_trace(
                go.Scatter(
                x=agg_data['instance_year'],
                y=agg_data['mean'],
                name="Avg Meter Sale Price",
                mode="lines+markers",
                line=dict(color='darkorange')
                ),
                secondary_y=False,
            )

            # Add bar plot for count (distribution)
            fig_combo.add_trace(
                go.Bar(
                x=agg_data['instance_year'],
                y=agg_data['count'],
                name="Count",
                marker_color='lightskyblue',
                opacity=0.6
                ),
                secondary_y=True,
            )

            # Set axis titles
            fig_combo.update_layout(
                xaxis_title="Instance Year",
                title="Avg Meter Sale Price and Count per Year",
                legend=dict(x=0.5, y=1.1, orientation='h', xanchor='center'),
            )

            fig_combo.update_yaxes(title_text="Avg Meter Sale Price", secondary_y=False)
            fig_combo.update_yaxes(title_text="Count", secondary_y=True)

            #    Display plot
            st.plotly_chart(fig_combo, use_container_width=True)



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
        clean_xls = pd.ExcelFile(cat_plot_path_clean)
        sheet_names = xls.sheet_names
        clean_sheet_names = clean_xls.sheet_names
    except FileNotFoundError:
        st.error(f"File not found: {cat_plot_path} or {cat_plot_path_clean}")
        st.stop()

    sheet = st.selectbox("Select Sheet to Visualize", sheet_names)
    df_plot = pd.read_excel(xls, sheet_name=sheet)

    if sheet in clean_sheet_names:
        df_clean_plot = pd.read_excel(clean_xls, sheet_name=sheet)
    else:
        st.warning(f"Sheet '{sheet}' not found in cleaned data. Showing only original.")
        df_clean_plot = None

    st.write(f"### Sheet: {sheet}")
    st.dataframe(df_plot)

    # --- Plotting Functions ---
    def plot_boxplot(df, title_suffix=""):
        if 'instance_year' not in df.columns:
            return None

        group_col = df.columns[2] if len(df.columns) > 2 else None
        required_cols = {'count', 'min', 'mean', '25%', '50%', '75%', 'max'}
        if not required_cols.issubset(df.columns):
            return None

        if group_col and group_col in df.columns:
            grouped = df.groupby(group_col).agg({
                'count': 'sum',
                'min': 'min',
                'mean': 'mean',
                '25%': 'mean',
                '50%': 'mean',
                '75%': 'mean',
                'max': 'max'
            }).reset_index()
        else:
            grouped = pd.DataFrame([{
                'count': df['count'].sum(),
                'min': df['min'].min(),
                'mean': df['mean'].mean(),
                '25%': df['25%'].mean(),
                '50%': df['50%'].mean(),
                '75%': df['75%'].mean(),
                'max': df['max'].max(),
                group_col: 'Overall'
            }])

        fig = go.Figure()
        colors = px.colors.qualitative.Set2

        for idx, (_, row) in enumerate(grouped.iterrows()):
            q1 = row['25%']
            q3 = row['75%']
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            name = row[group_col] if group_col else 'Overall'

            fig.add_trace(go.Box(
                name=name,
                y=[row['min'], q1, row['50%'], q3, row['max']],
                boxpoints='outliers',
                marker=dict(color=colors[idx % len(colors)]),
                line=dict(color=colors[idx % len(colors)]),
                q1=[q1],
                median=[row['50%']],
                q3=[q3],
                lowerfence=[lower_fence],
                upperfence=[upper_fence],
                orientation='v'
            ))

        fig.update_layout(
            title=f"Aggregated Box Plot {title_suffix} by {group_col if group_col else 'Overall'}",
            yaxis_title="Meter Sale Price",
            xaxis_title=group_col if group_col else '',
            boxmode='group'
        )
        return fig

    def plot_mean_line(df, title_suffix=""):
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
                    name=str(name)
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df['instance_year'],
                y=df['mean'],
                mode='lines+markers',
                name='Mean'
            ))

        fig.update_layout(
            title=f'Mean (Meter Sale Price) Over Years {title_suffix} by {legend_col if legend_col else "N/A"}',
            xaxis_title='Instance Year',
            yaxis_title='Mean (Meter Sale Price)',
            hovermode='x unified'
        )
        return fig

    # --- Original Data Plots (Side by Side) ---
    st.markdown("### 📊 Original Data Plots")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Aggregated Box Plot (Original)")
        box_fig = plot_boxplot(df_plot)
        if box_fig:
            st.plotly_chart(box_fig, use_container_width=True)
        else:
            st.info("Box plot not available due to missing columns or data.")

    with col2:
        st.subheader("📈 Mean Line Plot (Original)")
        line_fig = plot_mean_line(df_plot)
        if line_fig:
            st.plotly_chart(line_fig, use_container_width=True)
        else:
            st.info("Mean line plot not available due to missing columns or data.")

    # --- Cleaned Data Plots (Side by Side Below) ---
    if df_clean_plot is not None:
        st.markdown("### 🧹 Cleaned Data Plots")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("📦 Aggregated Box Plot (Cleaned)")
            box_fig_clean = plot_boxplot(df_clean_plot)
            if box_fig_clean:
                st.plotly_chart(box_fig_clean, use_container_width=True)
            else:
                st.info("Box plot not available due to missing columns or data.")

        with col4:
            st.subheader("📈 Mean Line Plot (Cleaned)")
            line_fig_clean = plot_mean_line(df_clean_plot)
            if line_fig_clean:
                st.plotly_chart(line_fig_clean, use_container_width=True)
            else:
                st.info("Mean line plot not available due to missing columns or data.")


################

# Define file paths
EXCEL_PATH = "All_model_output.xlsx"
html_lr = "predicted_vs_actual_linear.html"
html_dt = "predicted_vs_actual_decision_tree.html"
html_xgb = "predicted_vs_actual_XGB_regressor.html"
html_comparision = "model_performance_comparison.html"

# Load Excel file with caching
@st.cache_data
def load_excel(path):
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names
    data = {sheet: xls.parse(sheet) for sheet in sheets}
    return data

# Main view logic
if sidebar_option == "Model Output":
    st.header("Model Output")

    st.markdown("""
        Instead of segmenting the data by property type, we opted to model all property types together, with a primary focus on units.  
        **Time Frame:** The analysis includes data from the year 2020 onwards.  
        Given the large number of independent variables, we employed a stepwise regression approach to identify the most significant predictors for our model.  
        Using the variables selected through this process, we obtained the following results, primarily focused on unit-level data:
    """)

    if os.path.exists(EXCEL_PATH):
        sheet_data = load_excel(EXCEL_PATH)

        for sheet_name, df in sheet_data.items():
            st.subheader(f"Sheet: {sheet_name}")
            st.dataframe(df)

            if sheet_name == "all_model_overall_output":
                # Show comparison HTML report
                if os.path.exists(html_comparision):
                    with open(html_comparision, "r", encoding="utf-8") as f:
                        comparison_html = f.read()
                    st.markdown("### 🔎 Overall Comparison Report")
                    components.html(comparison_html, height=300, scrolling=True)
                else:
                    st.warning(f"Comparison HTML not found at: {html_comparision}")

                st.markdown("---")

                # Logistic Regression
                st.markdown("#### 📊 Logistic Regression")
                if os.path.exists(html_lr):
                    with open(html_lr, "r", encoding="utf-8") as f:
                        lr_html = f.read()
                    components.html(lr_html, height=400, scrolling=True)
                else:
                    st.warning(f"Logistic Regression HTML not found at: {html_lr}")

                # Decision Tree
                st.markdown("#### 🌳 Decision Tree")
                if os.path.exists(html_dt):
                    with open(html_dt, "r", encoding="utf-8") as f:
                        dt_html = f.read()
                    components.html(dt_html, height=400, scrolling=True)
                else:
                    st.warning(f"Decision Tree HTML not found at: {html_dt}")

                # XGBoost
                st.markdown("#### 🚀 XGBoost")
                if os.path.exists(html_xgb):
                    with open(html_xgb, "r", encoding="utf-8") as f:
                        xgb_html = f.read()
                    components.html(xgb_html, height=400, scrolling=True)
                else:
                    st.warning(f"XGBoost HTML not found at: {html_xgb}")

    else:
        st.error(f"Excel file not found at: {EXCEL_PATH}")
