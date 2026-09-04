# ============================================================
# SYSTEM CAPACITY & CARE LOAD ANALYTICS
# FOR UNACCOMPANIED CHILDREN (UAC)
#
# STREAMLIT DASHBOARD
# Compatible with:
# - Google Colab
# - Local Computer
# - GitHub
# - Streamlit Community Cloud
# ============================================================

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="UAC System Capacity Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 System Capacity & Care Load Analytics")
st.subheader("Unaccompanied Children (UAC) Care System")

st.markdown(
    """
    This dashboard analyzes system capacity, care load, intake pressure,
    backlog trends, and capacity strain across CBP and HHS care systems.
    """
)


# ============================================================
# DATASET PATH DETECTION
# ============================================================

FILE_NAME = "HHS_Unaccompanied_Alien_Children_Program.csv"

COLAB_PATH = os.path.join("/content", FILE_NAME)
LOCAL_PATH = FILE_NAME

if os.path.exists(COLAB_PATH):
    FILE_PATH = COLAB_PATH

elif os.path.exists(LOCAL_PATH):
    FILE_PATH = LOCAL_PATH

else:
    st.error(
        f"Dataset '{FILE_NAME}' was not found. "
        "Please make sure the CSV file is in the same folder as app.py."
    )
    st.stop()


# ============================================================
# LOAD AND PROCESS DATA
# ============================================================

@st.cache_data
def load_and_process_data(file_path):

    # Load dataset
    data = pd.read_csv(file_path)

    # Remove completely empty rows
    data = data.dropna(how="all").copy()

    # Clean column names
    data.columns = data.columns.astype(str).str.strip()

    # Rename columns
    column_mapping = {
        "Children apprehended and placed in CBP custody*":
            "Apprehended_CBP",

        "Children apprehended and placed in CBP custody":
            "Apprehended_CBP",

        "Children in CBP custody":
            "CBP_Custody",

        "Children transferred out of CBP custody":
            "Transferred_to_HHS",

        "Children in HHS Care":
            "HHS_Care",

        "Children discharged from HHS Care":
            "Discharged_HHS"
    }

    data = data.rename(columns=column_mapping)

    # Required columns
    required_columns = [
        "Date",
        "Apprehended_CBP",
        "CBP_Custody",
        "Transferred_to_HHS",
        "HHS_Care",
        "Discharged_HHS"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Convert Date column
    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    # Remove invalid dates
    data = data.dropna(
        subset=["Date"]
    ).copy()

    # Numeric columns
    numeric_columns = [
        "Apprehended_CBP",
        "CBP_Custody",
        "Transferred_to_HHS",
        "HHS_Care",
        "Discharged_HHS"
    ]

    # Convert numeric values
    for column in numeric_columns:

        data[column] = (
            data[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        data[column] = (
            data[column]
            .fillna(0)
            .clip(lower=0)
        )

    # Sort by Date
    data = data.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    # Remove duplicate dates
    data = data.drop_duplicates(
        subset=["Date"],
        keep="last"
    ).copy()


    # ========================================================
    # DATA QUALITY METRICS
    # ========================================================

    data["Transfer_Anomaly"] = (
        data["Transferred_to_HHS"]
        >
        data["CBP_Custody"]
    )

    data["Discharge_Anomaly"] = (
        data["Discharged_HHS"]
        >
        data["HHS_Care"]
    )

    data["Reporting_Anomaly"] = (
        data["Transfer_Anomaly"]
        |
        data["Discharge_Anomaly"]
    )


    # ========================================================
    # DERIVED METRICS
    # ========================================================

    # Total System Load
    data["Total_System_Load"] = (
        data["CBP_Custody"]
        +
        data["HHS_Care"]
    )

    # Net Daily Intake
    data["Net_Daily_Intake"] = (
        data["Transferred_to_HHS"]
        -
        data["Discharged_HHS"]
    )

    # Care Load Growth Rate
    data["Care_Load_Growth_Rate"] = (
        data["Total_System_Load"]
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        * 100
    )

    # Backlog Indicator
    data["Backlog_Indicator"] = (
        data["Net_Daily_Intake"]
        .cumsum()
    )

    # Discharge Offset Ratio
    data["Discharge_Offset_Ratio"] = np.where(
        data["Transferred_to_HHS"] > 0,
        (
            data["Discharged_HHS"]
            /
            data["Transferred_to_HHS"]
        ),
        0
    )

    # Rolling averages
    data["System_Load_7D_Avg"] = (
        data["Total_System_Load"]
        .rolling(
            window=7,
            min_periods=1
        )
        .mean()
    )

    data["System_Load_14D_Avg"] = (
        data["Total_System_Load"]
        .rolling(
            window=14,
            min_periods=1
        )
        .mean()
    )

    data["Net_Intake_7D_Avg"] = (
        data["Net_Daily_Intake"]
        .rolling(
            window=7,
            min_periods=1
        )
        .mean()
    )

    # Daily Load Change
    data["Daily_Load_Change"] = (
        data["Total_System_Load"]
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # Care Load Volatility
    data["Care_Load_Volatility"] = (
        data["Daily_Load_Change"]
        .rolling(
            window=7,
            min_periods=2
        )
        .std()
        .fillna(0)
        * 100
    )

    # High Load Threshold
    high_load_threshold = (
        data["Total_System_Load"]
        .quantile(0.75)
    )

    data["High_Load_Threshold"] = (
        high_load_threshold
    )

    data["High_Load"] = (
        data["Total_System_Load"]
        >= high_load_threshold
    )

    # Capacity Strain
    data["Capacity_Strain"] = (
        data["High_Load"]
        &
        (
            data["Net_Daily_Intake"]
            > 0
        )
    )

    # Capacity Relief
    data["Capacity_Relief"] = (
        data["Net_Daily_Intake"]
        < 0
    )

    # Time Features
    data["Year"] = (
        data["Date"]
        .dt.year
    )

    data["Month"] = (
        data["Date"]
        .dt.month
    )

    data["Month_Name"] = (
        data["Date"]
        .dt.month_name()
    )

    return data


# ============================================================
# LOAD DATA SAFELY
# ============================================================

try:

    df = load_and_process_data(
        FILE_PATH
    )

except Exception as error:

    st.error(
        f"Error while processing the dataset: {error}"
    )

    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("📅 Dashboard Filters")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)


# Handle selected date range
if isinstance(selected_dates, tuple) or isinstance(selected_dates, list):

    if len(selected_dates) == 2:

        start_date = pd.to_datetime(
            selected_dates[0]
        )

        end_date = pd.to_datetime(
            selected_dates[1]
        )

    else:

        start_date = pd.to_datetime(
            min_date
        )

        end_date = pd.to_datetime(
            max_date
        )

else:

    start_date = pd.to_datetime(
        min_date
    )

    end_date = pd.to_datetime(
        max_date
    )


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df[
    (
        df["Date"]
        >= start_date
    )
    &
    (
        df["Date"]
        <= end_date
    )
].copy()


if filtered_df.empty:

    st.warning(
        "No data is available for the selected date range."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_children_under_care = (
    filtered_df["Total_System_Load"]
    .iloc[-1]
)

average_net_intake = (
    filtered_df["Net_Daily_Intake"]
    .mean()
)

volatility_index = (
    filtered_df["Care_Load_Volatility"]
    .mean()
)

backlog_rate = (
    (
        filtered_df["Net_Daily_Intake"]
        > 0
    )
    .mean()
    * 100
)

discharge_offset_ratio = (
    filtered_df["Discharge_Offset_Ratio"]
    .mean()
    * 100
)

strain_rate = (
    filtered_df["Capacity_Strain"]
    .mean()
    * 100
)


# ============================================================
# KPI SUMMARY CARDS
# ============================================================

st.header("📌 KPI Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Children Under Care",
    f"{total_children_under_care:,.0f}"
)

col2.metric(
    "Average Net Intake",
    f"{average_net_intake:,.2f}"
)

col3.metric(
    "Care Load Volatility",
    f"{volatility_index:,.2f}%"
)


col4, col5, col6 = st.columns(3)

col4.metric(
    "Backlog Accumulation Rate",
    f"{backlog_rate:,.2f}%"
)

col5.metric(
    "Discharge Offset Ratio",
    f"{discharge_offset_ratio:,.2f}%"
)

col6.metric(
    "Capacity Strain Rate",
    f"{strain_rate:,.2f}%"
)


# ============================================================
# SYSTEM LOAD OVERVIEW
# ============================================================

st.header("📈 System Load Overview")

fig_system_load = go.Figure()

fig_system_load.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Total_System_Load"],
        mode="lines",
        name="Total System Load"
    )
)

fig_system_load.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["System_Load_7D_Avg"],
        mode="lines",
        name="7-Day Average"
    )
)

fig_system_load.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["System_Load_14D_Avg"],
        mode="lines",
        name="14-Day Average"
    )
)

fig_system_load.update_layout(
    title="Total System Load Over Time",
    xaxis_title="Date",
    yaxis_title="Number of Children",
    hovermode="x unified"
)

st.plotly_chart(
    fig_system_load,
    use_container_width=True
)


# ============================================================
# CBP VS HHS LOAD
# ============================================================

st.header("🔄 CBP vs HHS Care Load")

fig_cbp_hhs = go.Figure()

fig_cbp_hhs.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["CBP_Custody"],
        mode="lines",
        name="CBP Custody"
    )
)

fig_cbp_hhs.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["HHS_Care"],
        mode="lines",
        name="HHS Care"
    )
)

fig_cbp_hhs.update_layout(
    title="CBP Custody vs HHS Care",
    xaxis_title="Date",
    yaxis_title="Number of Children",
    hovermode="x unified"
)

st.plotly_chart(
    fig_cbp_hhs,
    use_container_width=True
)


# ============================================================
# NET INTAKE ANALYSIS
# ============================================================

st.header("📉 Net Intake Pressure")

fig_net_intake = go.Figure()

fig_net_intake.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Net_Daily_Intake"],
        mode="lines",
        name="Net Daily Intake"
    )
)

fig_net_intake.add_hline(
    y=0,
    line_dash="dash"
)

fig_net_intake.update_layout(
    title="Transfers to HHS Minus Discharges",
    xaxis_title="Date",
    yaxis_title="Net Daily Intake",
    hovermode="x unified"
)

st.plotly_chart(
    fig_net_intake,
    use_container_width=True
)


# ============================================================
# BACKLOG TREND
# ============================================================

st.header("📊 Backlog Accumulation")

fig_backlog = px.line(
    filtered_df,
    x="Date",
    y="Backlog_Indicator",
    title="Backlog Accumulation Trend"
)

fig_backlog.update_layout(
    xaxis_title="Date",
    yaxis_title="Cumulative Net Intake"
)

st.plotly_chart(
    fig_backlog,
    use_container_width=True
)


# ============================================================
# CAPACITY STRAIN ANALYSIS
# ============================================================

st.header("🚨 Capacity Strain Analysis")

fig_strain = go.Figure()

fig_strain.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Total_System_Load"],
        mode="lines",
        name="Total System Load"
    )
)

strain_data = filtered_df[
    filtered_df["Capacity_Strain"]
].copy()

if not strain_data.empty:

    fig_strain.add_trace(
        go.Scatter(
            x=strain_data["Date"],
            y=strain_data["Total_System_Load"],
            mode="markers",
            name="Capacity Strain"
        )
    )

fig_strain.update_layout(
    title="Periods of Capacity Strain",
    xaxis_title="Date",
    yaxis_title="Total System Load",
    hovermode="x unified"
)

st.plotly_chart(
    fig_strain,
    use_container_width=True
)


# ============================================================
# MONTHLY ANALYSIS
# ============================================================

st.header("📅 Monthly Average System Load")

monthly_data = (
    filtered_df
    .set_index("Date")
    .resample("ME")["Total_System_Load"]
    .mean()
    .reset_index()
)

fig_monthly = px.line(
    monthly_data,
    x="Date",
    y="Total_System_Load",
    title="Monthly Average System Load"
)

fig_monthly.update_layout(
    xaxis_title="Month",
    yaxis_title="Average System Load"
)

st.plotly_chart(
    fig_monthly,
    use_container_width=True
)


# ============================================================
# DATA QUALITY SECTION
# ============================================================

st.header("🔍 Data Quality Summary")

quality_col1, quality_col2, quality_col3 = st.columns(3)

quality_col1.metric(
    "Transfer Anomalies",
    int(
        filtered_df[
            "Transfer_Anomaly"
        ].sum()
    )
)

quality_col2.metric(
    "Discharge Anomalies",
    int(
        filtered_df[
            "Discharge_Anomaly"
        ].sum()
    )
)

quality_col3.metric(
    "Reporting Anomalies",
    int(
        filtered_df[
            "Reporting_Anomaly"
        ].sum()
    )
)


# ============================================================
# DATA PREVIEW
# ============================================================

st.header("📋 Analyzed Data Preview")

st.dataframe(
    filtered_df,
    use_container_width=True
)


# ============================================================
# DOWNLOAD ANALYZED DATA
# ============================================================

st.header("⬇️ Download Analyzed Data")

csv_data = (
    filtered_df
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="Download Analyzed UAC Data",
    data=csv_data,
    file_name="UAC_Analyzed_Data.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "System Capacity & Care Load Analytics for "
    "Unaccompanied Children (UAC)"
)
