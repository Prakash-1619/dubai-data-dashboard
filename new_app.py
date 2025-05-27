import streamlit as st
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
    "Plots on Categorical Columns"
])

# --- View 1: Data Preview ---
if sidebar_option == "Data Preview":
    tab1, tab2, tab3 = st.tabs(["Preview", "Summary", "Box Plots"])

    with tab1:
        sample_df = pd.read_csv(sample)
        st.subheader("📄 Original DF Preview")
        st.dataframe(sample_df)

    with tab2:
        st.subheader("📋 Data Summary for Original DF")
        summary_df = pd.read_excel(summary)
        st.dataframe(summary_df)


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
        sheet_names = xls.sheet_names
    except FileNotFoundError:
        st.error(f"File not found: {cat_plot_path}")
        st.stop()

    sheet = st.selectbox("Select Sheet to Visualize", sheet_names)
    df_plot = pd.read_excel(xls, sheet_name=sheet)
    st.write(f"### Sheet: {sheet}")
    st.dataframe(df_plot)

    # -------------- Raw Box Plot (with outliers) ----------------
    def plot_raw_boxplot(df):
        if 'meter_sale_price' not in df.columns or 'instance_year' not in df.columns:
            return None

        group_col = df.columns[1] if len(df.columns) > 2 else None
        fig = go.Figure()

        if group_col:
            for group, group_df in df.groupby(group_col):
                fig.add_trace(go.Box(
                    y=group_df['meter_sale_price'],
                    name=str(group),
                    boxpoints='outliers',  # Show outliers
                    marker=dict(color='rgba(0,0,255,0.5)'),
                    line=dict(color='blue')
                ))
        else:
            fig.add_trace(go.Box(
                y=df['meter_sale_price'],
                name='Overall',
                boxpoints='outliers'
            ))

        fig.update_layout(
            title='Box Plot with Outliers',
            yaxis_title='Meter Sale Price',
            boxmode='group'
        )
        return fig

    # -------------- Mean Line Plot ----------------
    def plot_mean_line(df):
        if 'instance_year' not in df.columns or 'meter_sale_price' not in df.columns:
            return None

        legend_col = df.columns[1] if len(df.columns) > 2 else None
        fig = go.Figure()

        if legend_col and legend_col in df.columns:
            grouped = df.groupby([legend_col, 'instance_year'])['meter_sale_price'].mean().reset_index()
            for name in grouped[legend_col].unique():
                group_df = grouped[grouped[legend_col] == name]
                fig.add_trace(go.Scatter(
                    x=group_df['instance_year'],
                    y=group_df['meter_sale_price'],
                    mode='lines+markers',
                    name=f'{name} - Mean'
                ))
        else:
            grouped = df.groupby('instance_year')['meter_sale_price'].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=grouped['instance_year'],
                y=grouped['meter_sale_price'],
                mode='lines+markers',
                name='Mean'
            ))

        fig.update_layout(
            title=f'Mean Meter Sale Price Over Years',
            xaxis_title='Instance Year',
            yaxis_title='Mean Meter Sale Price',
            hovermode='x unified'
        )
        return fig

    # -------------- Streamlit Layout ----------------
    if 'meter_sale_price' not in df_plot.columns or 'instance_year' not in df_plot.columns:
        st.warning("Required columns ('meter_sale_price', 'instance_year') not found in the dataset.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📦 Box Plot with Outliers")
            box_fig = plot_raw_boxplot(df_plot)
            if box_fig:
                st.plotly_chart(box_fig, use_container_width=True)
            else:
                st.info("Box plot not available due to missing data.")

        with col2:
            st.subheader("📈 Mean Line Plot")
            line_fig = plot_mean_line(df_plot)
            if line_fig:
                st.plotly_chart(line_fig, use_container_width=True)
            else:
                st.info("Mean line plot not available.")

