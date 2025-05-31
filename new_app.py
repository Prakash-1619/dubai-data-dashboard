import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- Page Config ---
st.set_page_config(layout="wide")
st.sidebar.title("🔍 Flipose-RE-Analytics")
# --- File Paths ---
df_path = "target_df.csv"
area_stats_path = "df_area_plot_stats.xlsx"
summary = "data_summary.xlsx"
sample = "sample_df.csv"

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

st.sidebar.success("All data loaded, 🔍 Explore the Dash Board")

# --- Load Area Stats ---
df_area_plot_stats = load_excel(area_stats_path)

# --- Sidebar Navigation ---
sidebar_option = st.sidebar.radio("." ,[
    "Data Summary",    
    "Pareto Analysis",
    "Univariate Analysis",
    "Bivariate Analysis",
    "Geo Graphical Analysis",
    "Price Prediction Model"
])

# --- View 1: Data Preview ---
if sidebar_option == "Data Summary":
    st.subheader("📄 Transactions Data")
    tab1, tab2 = st.tabs(["Data Preview", "Summary"])

    with tab1:
        sample_df = pd.read_csv(sample)
        st.markdown("--> Repeated columns i.e Arabic and Id columns are dropped from Data")
        sample_df  = sample_df.drop(sample_df.columns[0], axis=1)
        st.dataframe(sample_df)

    with tab2:
        col1, col2,col3,col4 = st.columns(4)
        with col1:
            st.metric(label="Num of Columns", value=46)
        with col2:
            st.metric(label="Total Records", value="1,424,588")
        with col3:
            st.metric(label="Start Date(Instance_date)", value="1966-01-18")
        with col4:
            st.metric(label="End date(Instance_date)", value="2025-04-03")
        
        summary_df = pd.read_excel(summary)
        # Format all numeric columns with commas
        for col in summary_df.select_dtypes(include='number').columns:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)

        summary_df.index = range(1, len(summary_df) + 1)
        summary_df.rename(columns={'No_of_units': 'Num_of_Unique_values'}, inplace=True)
        summary_df = summary_df.drop(columns = ["S.no", "Level"])
        st.dataframe(summary_df)


###########################################################################################################################################
elif sidebar_option == "Pareto Analysis":
    st.subheader("📋 Pareto Analysis")
    try:
            pereto_file = "pereto_analysis_file.xlsx"
            html_pereto_df = "pareto_analysis_plot.html"
            #html_pereto_df_clean = "pareto_analysis_plot_after_model_run.html"
            pereto_analyis = pd.ExcelFile(pereto_file)
            pereto_sheet_names = pereto_analyis.sheet_names

    except FileNotFoundError:
            st.error(f"File not found: {pereto_file}")
            st.stop()


    all_sheets_df = pd.read_excel(pereto_analyis, sheet_name=pereto_sheet_names)

    # Extract specific sheets
    pareto_summary = all_sheets_df["Pereto_Analysis_by_area_name"]
    ABC_summary = all_sheets_df["ABC_Area_name"]

    tab1, tab2 = st.tabs(["ABC Summary", "Pareto Analysis Table"])

    with tab1:
        st.markdown("### 📊 Pareto Analysis Plot")
    
        if os.path.exists(html_pereto_df):
            with open(html_pereto_df, "r", encoding="utf-8") as f:
                dt_html = f.read()
            components.html(dt_html, height=500, width=3500, scrolling=False)
        else:
            st.error("Pareto analysis HTML file not found.")

        # ABC Summary Table
        st.markdown("### 📋 ABC Summary Table")
        if ABC_summary is not None and not ABC_summary.empty:
            if 'nRecords' in ABC_summary.columns:
                ABC_summary['nRecords'] = ABC_summary['nRecords'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
            ABC_summary.index = range(1, len(ABC_summary) + 1)
            st.dataframe(ABC_summary, use_container_width=True)
        else:
            st.warning("ABC Summary data is empty or not loaded.")


    with tab2:
        st.markdown("### Pareto Analysis by Area_name_en")
        pareto_summary['nRecords'] =  pareto_summary['nRecords'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
        pareto_summary.index = range(1, len(pareto_summary) + 1)
        st.dataframe(pareto_summary, use_container_width=True)



#########################################################################################################################################################
if sidebar_option == "Geo Graphical Analysis":
    st.subheader("Dubai Area-wise Bubble Map")
    df_excel = pd.read_excel("new_tdf.xlsx")
    units_excel = pd.read_excel("units_20.xlsx")
    # Create first bubble map 
    Count_tab, Avg_tab = st.tabs(["nRecords","Average Meter sale price"])
    with Count_tab:
        figs = px.scatter_mapbox(
            df_excel,
            lat='area_lat',
            lon='area_lon',
            size='Average Meter Sale Price',
            color='Transaction Count',
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

        for trace in figs.data:
            trace.name = "Raw data"
            trace.legendgroup = "Raw data"
            trace.showlegend = True
    
        # Create second bubble map
        fig2 = px.scatter_mapbox(
            units_excel,
            lat='area_lat',
            lon='area_lon',
            size='Average Meter Sale Price',
            color='Transaction Count',
            hover_name='area_name_en',
            hover_data={
                'Transaction Count': True,
                'Average Meter Sale Price': ':.2f',
                'area_lat': False,
                'area_lon': False
            },
            color_continuous_scale='Viridis',
            size_max=30,
            opacity=0.6,
            zoom=9,
            title="Dubai Area-wise Average Meter Sale Price and Transaction Count"
        )

        for trace in fig2.data:
            trace.name = "Data >= 2020"
            trace.legendgroup = "Data >= 2020"
            trace.showlegend = True

        # Combine the two maps
        for trace in fig2.data:
            figs.add_trace(trace)

        figs.update_layout(
            mapbox_style='open-street-map',
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )

        # Display the map
        st.plotly_chart(figs, use_container_width=True)

    
    with Avg_tab:
        figs = px.scatter_mapbox(
            df_excel,
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

        for trace in figs.data:
            trace.name = "Raw data"
            trace.legendgroup = "Raw data"
            trace.showlegend = True
    
        # Create second bubble map
        fig2 = px.scatter_mapbox(
            units_excel,
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
            opacity=0.6,
            zoom=9,
            title="Dubai Area-wise Average Meter Sale Price and Transaction Count"
        )

        for trace in fig2.data:
            trace.name = "Data >= 2020"
            trace.legendgroup = "Data >= 2020"
            trace.showlegend = True

        # Combine the two maps
        for trace in fig2.data:
            figs.add_trace(trace)

        figs.update_layout(
            mapbox_style='open-street-map',
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )

        # Display the map
        st.plotly_chart(figs, use_container_width=True)


############################################################################################################################################################
# --- View 3: Plots on Categorical Columns --

def calculate_boxplot_stats(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    median = series.quantile(0.5)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    min_val = series.min()
    max_val = series.max()

    return {
        'Min': min_val,
        'Q1': Q1,
        'Median': median,
        'Q3': Q3,
        'Max': max_val,
        'IQR': IQR,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Outliers': outliers
    }

# --- Load Excel ---
try:
    cat_plot_path = "original_df_description_tables.xlsx
    cat_plot_path_clean = "original_df_description_tables.xlsx
    xls = pd.ExcelFile(cat_plot_path)
    clean_xls = pd.ExcelFile(cat_plot_path_clean)
    sheet_names = xls.sheet_names
    clean_sheet_names = clean_xls.sheet_names
except FileNotFoundError:
    st.error("File not found.")
    st.stop()

sheet = st.sidebar.selectbox("Select Sheet", sheet_names)
df_plot = pd.read_excel(xls, sheet_name=sheet)
df_clean_plot = pd.read_excel(clean_xls, sheet_name=sheet) if sheet in clean_sheet_names else None

numeric_cols = df_plot.select_dtypes(include='number').columns.tolist()
selected_column = st.sidebar.selectbox("Select Numeric Column", numeric_cols)
cat_column = st.sidebar.selectbox("Select Categorical Column (x-axis)", df_plot.select_dtypes(include='object').columns.tolist())

# --- TABS ---
tab1, tab2 = st.tabs(["📋 Summary Table", "📦 Box + Distribution Plots"])

# --- TAB 1: SUMMARY ---
with tab1:
    st.markdown(f"### Summary Stats for `{selected_column}`")
    stats = calculate_boxplot_stats(df_plot[selected_column].dropna())
    st.write(pd.DataFrame.from_dict(stats, orient='index', columns=['Value']))

# --- TAB 2: PLOTS ---
with tab2:
    st.markdown("### Distribution Plot (Original)")
    fig_dist, ax = plt.subplots()
    sns.boxplot(data=df_plot, x=cat_column, y=selected_column, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig_dist)

    st.markdown("### Custom Box Plot with IQR Logic")
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df_plot[selected_column],
        name="Original",
        boxpoints='outliers',
        marker_color='indianred'
    ))

    if df_clean_plot is not None:
        fig.add_trace(go.Box(
            y=df_clean_plot[selected_column],
            name="Cleaned",
            boxpoints='outliers',
            marker_color='seagreen'
        ))

    fig.update_layout(
        title=f"Box Plot for {selected_column}",
        yaxis_title=selected_column,
        boxmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)





##################################################################################################################

# Define file paths
EXCEL_PATH = "All_model_output.xlsx"
model_perfomance =  "Model_performance.xlsx"
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


# === Sidebar Selection ===
if sidebar_option == "Price Prediction Model":

    # === Top-Level Tabs ===
    main_tabs = st.tabs(["📈 Model Performance Tables","📉 Prediction Model Visuals"])
    
    # === Tab 1: Prediction Model Visuals ===
    with main_tabs[1]:
        if os.path.exists(model_perfomance):
            xl = pd.ExcelFile(model_perfomance)
            sheet_names = xl.sheet_names

        if len(sheet_names) >= 2:
            second_sheet_name = sheet_names[1]  # Index 1 = second sheet
            df = xl.parse(sheet_name=second_sheet_name)
            df = df.round(2)
            if 'nRecords' in df.columns:
                df['nRecords'] = df['nRecords'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
            df.index = range(1, len(df) + 1)

            st.subheader(f"📊 {second_sheet_name}")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("The Excel file has less than 2 sheets.")
            
        st.subheader("🔍 Overall Comparison Report")
        if os.path.exists(html_comparision):
            with open(html_comparision, "r", encoding="utf-8") as f:
                components.html(f.read(), height=300, scrolling=True)
        else:
            st.warning(f"Comparison HTML not found at: {html_comparision}")

        st.subheader("📊 Logistic Regression")
        st.markdown("###Equation : Predicted_price = 0.40134 * Actual_price + 8966.97")
        if os.path.exists(html_lr):
            with open(html_lr, "r", encoding="utf-8") as f:
                components.html(f.read(), height=400, scrolling=True)
        else:
            st.warning(f"Logistic Regression HTML not found at: {html_lr}")

        st.subheader("🌳 Decision Tree")
        st.markdown("###Equation : Predicted_price = 0.465166 * Actual_price + 7993.22")
        if os.path.exists(html_dt):
            with open(html_dt, "r", encoding="utf-8") as f:
                components.html(f.read(), height=400, scrolling=True)
        else:
            st.warning(f"Decision Tree HTML not found at: {html_dt}")

        st.subheader("🚀 XGBoost")
        st.markdown("###Equation : Predicted_price = 0.463650 * Actual_price + 8055.86")
        if os.path.exists(html_xgb):
            with open(html_xgb, "r", encoding="utf-8") as f:
                components.html(f.read(), height=400, scrolling=True)
        else:
            st.warning(f"XGBoost HTML not found at: {html_xgb}")

    # === Tab 3: Area & Sector Sheets ===
    with main_tabs[0]:
        if os.path.exists(EXCEL_PATH):
            sheet_data_main = load_excel(EXCEL_PATH)
            
        Over_all, area_tab,sector_tab = st.tabs(["Over All","Area wise","Sector wise"])
        with Over_all:
            st.subheader("📍 Prediction Models Over All")
            if os.path.exists(model_perfomance):
                sheet_data = load_excel(model_perfomance)
                perf_tabs = st.tabs(list(sheet_data.keys()))
                for tab, (sheet_name, df) in zip(perf_tabs, sheet_data.items()):
                    with tab:
                        df = df.round(2)
                        if 'nRecords' in df.columns:
                            df['nRecords'] = df['nRecords'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
                        df.index = range(1, len(df) + 1)
                        st.dataframe(df, use_container_width=True)
            else:
                st.error(f"Model performance file not found at: {model_perfomance}")
        with area_tab:
            area_sheets = {name: df for name, df in sheet_data_main.items() if "area" in name.lower()}
            
            # Subtabs for Area
            if area_sheets:
                st.subheader("📍 Prediction Model by Area")
                area_tabs = st.tabs(list(area_sheets.keys()))
                for tab, (sheet_name, df) in zip(area_tabs, area_sheets.items()):
                    with tab:
                        df = df.round(2)
                        if 'nRecords' in df.columns:
                            df['nRecords'] = df['nRecords'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
                        df.index = range(1, len(df) + 1)
                        st.dataframe(df, use_container_width=True)    
        with sector_tab:
            sector_sheets = {name: df for name, df in sheet_data_main.items() if "sector" in name.lower()}
            # Subtabs for Sector
            if sector_sheets:
                st.subheader("🏗️ Prediction Model by Sector")
                sector_tabs = st.tabs(list(sector_sheets.keys()))
                for tab, (sheet_name, df) in zip(sector_tabs, sector_sheets.items()):
                    with tab:
                        df = df.round(2)
                        if 'nRecords' in df.columns:
                            df['nRecords'] = df['nRecords'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
                        df.index = range(1, len(df) + 1)
                        st.dataframe(df, use_container_width=True)
            else:
                st.error(f"Excel file not found at: {EXCEL_PATH}")
    



