import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import io
import os
from streamlit.components.v1 import html
from ydata_profiling import ProfileReport
import calendar

# Set page config
st.set_page_config(page_title="Exploratory Data Analysis App by Ashwin Nair", layout="wide")

def save_uploaded_file(uploadedfile, directory="tempDir"):
    """Save uploaded file temporarily."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

def load_data(file_path):
    """Load data from a CSV or Excel file."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Error in loading data: {str(e)}")
        return None

def load_and_preprocess_data(df, selected_column):
    """Load and preprocess the data from an uploaded file."""
    try:
        if df.empty or len(df.columns) <= 1:
            st.error("Unable to parse the CSV file. Please check the file format and try again.")
            return None
        
        st.write("Preview of the loaded data:")
        st.write(df.head())
        
        date_column = df.columns[0]
        
        date_formats = ['%d-%b-%y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']
        for fmt in date_formats:
            try:
                df[date_column] = pd.to_datetime(df[date_column], format=fmt)
                break
            except ValueError:
                continue
        else:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            if df[date_column].isnull().all():
                st.error("Date format not recognized. Please provide dates in a recognized format.")
                return None
        
        if selected_column not in df.columns:
            st.error(f"'{selected_column}' column not found in the dataset.")
            return None
        
        df[selected_column] = pd.to_numeric(df[selected_column], errors='coerce')
        if df[selected_column].isnull().any():
            st.warning(f"'{selected_column}' column contains non-numeric values. These will be treated as NaN.")
        
        df.set_index(date_column, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error in loading and preprocessing data: {str(e)}")
        return None

def extract_time_features(df):
    """Extract time-based features from the datetime index."""
    df['Year'] = df.index.year
    df['Quarter'] = df.index.quarter
    df['Month'] = df.index.month
    df['WeekOfYear'] = df.index.isocalendar().week
    df['WeekOfMonth'] = df.index.to_series().apply(lambda x: (x.day - 1) // 7 + 1)
    df['DayOfYear'] = df.index.dayofyear
    df['DayOfMonth'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfWeekName'] = df.index.day_name()
    df['IsWeekend'] = df['DayOfWeek'] >= 5
    df['WeekStart'] = df.index - pd.to_timedelta(df.index.weekday, unit='d')
    return df

def extract_fiscal_time_features(df, fiscal_calendar):
    """Extract fiscal time-based features."""
    # Merge data with fiscal calendar
    df = df.reset_index().merge(fiscal_calendar, how='left', on='Date').set_index('Date')

    # Extract fiscal features
    df['FiscalYear'] = df['FiscalYear']
    df['FiscalMonth'] = df['FiscalMonth']
    df['FiscalWeek'] = df['FiscalWeek']
    return df

def perform_time_series_analysis(df, selected_column):
    """Perform time series analysis including decomposition and outlier detection."""
    decomposition = seasonal_decompose(df[selected_column], model='additive', period=30)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid
    df['Outlier'] = np.abs(df['Residual']) > (2 * df['Residual'].std())
    return df, decomposition

def create_monthly_box_plots(df, selected_column, use_fiscal_calendar=False):
    """Create separate interactive box plots for each month of every year."""
    if use_fiscal_calendar:
        # Use fiscal months
        month_names = df['FiscalMonth'].unique()
        month_names.sort()
        month_label = 'FiscalMonth'
        year_label = 'FiscalYear'
    else:
        # Use normal calendar months
        month_names = range(1, 13)
        month_label = 'Month'
        year_label = 'Year'
    
    plots = []
    
    for month in month_names:
        monthly_data = df[df[month_label] == month]
        title_suffix = 'Fiscal ' if use_fiscal_calendar else ''
        fig = px.box(monthly_data, x=year_label, y=selected_column, title=f'{title_suffix}{calendar.month_name[month]} Box Plot')
        plots.append(fig)
    return plots

def calculate_fiscal_calendar_totals(df, fiscal_calendar, selected_column):
    """Calculate monthly and weekly totals based on the provided fiscal calendar."""
    
    # Verify fiscal calendar has necessary columns
    required_columns = {'Date', 'FiscalYear', 'FiscalMonth', 'FiscalWeek'}
    if not required_columns.issubset(fiscal_calendar.columns):
        st.error(f"Fiscal calendar is missing one or more required columns: {required_columns}")
        return None, None

    # Merge df with fiscal_calendar on date to get fiscal weeks and months
    df = df.reset_index().merge(fiscal_calendar, how='left', on='Date').set_index('Date')

    # Check for successful merge
    if 'FiscalYear' not in df.columns:
        st.error("Failed to merge fiscal calendar with data. Ensure date alignment is correct.")
        return None, None

    # Calculate fiscal monthly totals
    fiscal_monthly_totals = df.groupby(['FiscalYear', 'FiscalMonth'])[selected_column].sum()

    # Calculate fiscal weekly totals
    fiscal_weekly_totals = df.groupby(['FiscalYear', 'FiscalWeek'])[selected_column].sum()

    # Convert series to DataFrame for display
    fiscal_monthly_totals_df = fiscal_monthly_totals.unstack()
    fiscal_weekly_totals_df = fiscal_weekly_totals.unstack()

    return fiscal_monthly_totals_df, fiscal_weekly_totals_df

def load_fiscal_calendar(uploaded_file):
    """Load the fiscal calendar from an Excel file."""
    try:
        fiscal_calendar = pd.read_excel(uploaded_file)
        fiscal_calendar['Date'] = pd.to_datetime(fiscal_calendar['Date'])

        # Ensure all necessary columns are present
        if not {'FiscalYear', 'FiscalMonth', 'FiscalWeek'}.issubset(fiscal_calendar.columns):
            st.error("Fiscal calendar file must contain columns: FiscalYear, FiscalMonth, FiscalWeek.")
            return None

        return fiscal_calendar
    except Exception as e:
        st.error(f"Error in loading fiscal calendar: {str(e)}")
        return None

def main():
    st.title("Exploratory Data Analysis App by Ashwin Nair")

    # Sidebar for file upload and navigation
    st.sidebar.title("Upload your CSV or Excel file")
    uploaded_file = st.sidebar.file_uploader("Drag and drop your data file here", type=["csv", "xlsx"])

    st.sidebar.title("Upload Fiscal Calendar")
    fiscal_calendar_file = st.sidebar.file_uploader("Upload your fiscal calendar file here", type=["xlsx"])

    st.sidebar.title("Navigation")
    st.sidebar.markdown("### Choose a Page", unsafe_allow_html=True)
    pages = ["Profiling", "Data Exploration", "Time Series Analysis", "Seasonality Analysis", "Monthly Box Plots", "Totals Analysis", "Fiscal Calendar Totals"]
    page = st.sidebar.radio("", pages, key='pages')

    if uploaded_file is not None and fiscal_calendar_file is not None:
        # Save files
        data_file_path = save_uploaded_file(uploaded_file)
        fiscal_calendar_file_path = save_uploaded_file(fiscal_calendar_file)

        # Load data
        df = load_data(data_file_path)

        # Load fiscal calendar
        fiscal_calendar = load_fiscal_calendar(fiscal_calendar_file_path)

        if fiscal_calendar is None:
            st.error("Failed to load fiscal calendar.")
            return

        if page == "Profiling":
            st.title("Profiling App by Ashwin")
            
            if df is not None:
                profile = ProfileReport(df, title="Profiling Report")
                profile.to_file("report.html")
                
                with open("report.html", "r", encoding='utf-8') as f:
                    report_html = f.read()
                
                html(report_html, width=1200, height=800, scrolling=True)

        else:
            if df is not None:
                columns = df.columns.tolist()
                date_column = columns.pop(0)
                
                selected_column = st.selectbox("Select the data column to analyze", columns)
                
                df = load_and_preprocess_data(df, selected_column)
                
                if df is not None:
                    # Sidebar option for calendar type selection
                    calendar_type = st.sidebar.radio("Select Calendar Type:", ("Normal Calendar", "Fiscal Calendar"))

                    # Extract time features based on selected calendar type
                    if calendar_type == "Normal Calendar":
                        df = extract_time_features(df)
                    else:
                        df = extract_fiscal_time_features(df, fiscal_calendar)
                    
                    df, decomposition = perform_time_series_analysis(df, selected_column)

                    if page == "Data Exploration":
                        st.header("**Data Exploration**")
                        st.write(df.describe())

                    elif page == "Time Series Analysis":
                        st.header("**Time Series Analysis**")
                        st.subheader("**Time Series of Selected Data**")
                        mean_value = df[selected_column].mean()
                        std_dev = df[selected_column].std()
                        ucl1 = mean_value + 1 * std_dev
                        lcl1 = mean_value - 1 * std_dev
                        ucl2 = mean_value + 2 * std_dev
                        lcl2 = mean_value - 2 * std_dev
                        ucl3 = mean_value + 3 * std_dev
                        lcl3 = mean_value - 3 * std_dev

                        # Plotting the time series with control limits
                        fig = px.line(df, x=df.index, y=selected_column, title=f'Time Series of {selected_column}')
                        fig.add_trace(go.Scatter(x=df.index, y=[ucl1] * len(df), mode='lines', name='UCL 1 STD', line=dict(color='green', dash='dash')))
                        fig.add_trace(go.Scatter(x=df.index, y=[ucl2] * len(df), mode='lines', name='UCL 2 STD', line=dict(color='yellow', dash='dash')))
                        fig.add_trace(go.Scatter(x=df.index, y=[ucl3] * len(df), mode='lines', name='UCL 3 STD', line=dict(color='red', dash='dash')))
                        fig.add_trace(go.Scatter(x=df.index, y=[mean_value] * len(df), mode='lines', name='Mean', line=dict(color='blue', dash='dash')))
                        fig.add_trace(go.Scatter(x=df.index, y=[lcl1] * len(df), mode='lines', name='LCL 1 STD', line=dict(color='green', dash='dash')))
                        fig.add_trace(go.Scatter(x=df.index, y=[lcl2] * len(df), mode='lines', name='LCL 2 STD', line=dict(color='yellow', dash='dash')))
                        fig.add_trace(go.Scatter(x=df.index, y=[lcl3] * len(df), mode='lines', name='LCL 3 STD', line=dict(color='red', dash='dash')))
                        fig.add_trace(go.Scatter(x=df[df['Outlier']].index, y=df[df['Outlier']][selected_column], mode='markers', marker=dict(color='orange', size=6), name='Outliers'))
                        st.plotly_chart(fig)

                        # Decomposition plots
                        st.subheader("**Time Series Decomposition**")
                        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
                        fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
                        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
                        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)
                        fig.update_layout(height=800, title_text="Time Series Decomposition")
                        st.plotly_chart(fig)

                    elif page == "Seasonality Analysis":
                        st.header("**Seasonality Analysis**")

                        # Adjust filtering options based on calendar type
                        if calendar_type == "Normal Calendar":
                            filter_option = st.radio("Filter by:", ("No Filter", "Year", "Month"))
                            if filter_option == "Year":
                                selected_year = st.selectbox("Select year:", df['Year'].unique())
                                filtered_df = df[df['Year'] == selected_year]
                            elif filter_option == "Month":
                                selected_month = st.selectbox("Select month:", df['Month'].unique())
                                filtered_df = df[df['Month'] == selected_month]
                            else:
                                filtered_df = df

                            seasonality_fields = ['Year', 'Quarter', 'Month', 'WeekOfYear', 'WeekOfMonth', 'DayOfYear', 'DayOfMonth', 'DayOfWeekName']
                        else:
                            filter_option = st.radio("Filter by:", ("No Filter", "Fiscal Year", "Fiscal Month"))
                            if filter_option == "Fiscal Year":
                                selected_year = st.selectbox("Select fiscal year:", df['FiscalYear'].unique())
                                filtered_df = df[df['FiscalYear'] == selected_year]
                            elif filter_option == "Fiscal Month":
                                selected_month = st.selectbox("Select fiscal month:", df['FiscalMonth'].unique())
                                filtered_df = df[df['FiscalMonth'] == selected_month]
                            else:
                                filtered_df = df

                            seasonality_fields = ['FiscalYear', 'FiscalMonth', 'FiscalWeek']

                        # Plot seasonality
                        for field in seasonality_fields:
                            fig = make_subplots(rows=1, cols=2, subplot_titles=(f'{field} Seasonality - Violin Plot', f'{field} Seasonality - Box Plot'))
                            violin = px.violin(filtered_df.reset_index(), x=field, y=selected_column, title=f'{field} Seasonality')
                            box = px.box(filtered_df.reset_index(), x=field, y=selected_column, title=f'{field} Seasonality')
                            
                            for trace in violin['data']:
                                fig.add_trace(trace, row=1, col=1)
                            
                            for trace in box['data']:
                                fig.add_trace(trace, row=1, col=2)

                            fig.update_layout(height=500, showlegend=False)
                            st.plotly_chart(fig)

                    elif page == "Monthly Box Plots":
                        st.header("**Monthly Box Plots**")
                        use_fiscal_calendar = calendar_type == "Fiscal Calendar"
                        box_plot_figs = create_monthly_box_plots(df, selected_column, use_fiscal_calendar)
                        for box_plot_fig in box_plot_figs:
                            st.plotly_chart(box_plot_fig)

                    elif page == "Totals Analysis":
                        st.header("**Monthly Totals**")
                        numeric_columns = df.select_dtypes(include=[np.number]).columns

                        # Correct monthly totals calculation
                        monthly_totals = df.resample('M').sum(numeric_only=True)
                        st.write("Monthly Totals (Tabular Format):")
                        st.write(monthly_totals[[selected_column]])
                        
                        # Create pivot table for monthly totals
                        monthly_totals_pivot = monthly_totals.pivot_table(index=monthly_totals.index.year, columns=monthly_totals.index.month, values=selected_column, aggfunc='sum')
                        monthly_totals_pivot.columns = [calendar.month_name[i] for i in monthly_totals_pivot.columns]
                        st.write("Monthly Totals (Pivot Table Format):")
                        st.write(monthly_totals_pivot)

                        st.header("**Weekly Totals**")
                        
                        # Correct weekly totals calculation
                        weekly_totals = df.resample('W').sum(numeric_only=True)
                        weekly_totals['WeekStart'] = weekly_totals.index - pd.to_timedelta(weekly_totals.index.weekday, unit='d')
                        weekly_totals.set_index('WeekStart', inplace=True)
                        st.write("Weekly Totals (Tabular Format):")
                        st.write(weekly_totals[[selected_column]])
                        
                        # Create pivot table for weekly totals
                        weekly_totals_pivot = weekly_totals.pivot_table(index=weekly_totals.index.year, columns=weekly_totals.index.isocalendar().week, values=selected_column, aggfunc='sum')
                        st.write("Weekly Totals (Pivot Table Format):")
                        st.write(weekly_totals_pivot)

                        st.header("Export Processed Data")
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, sheet_name='Processed Data')
                            monthly_totals.to_excel(writer, sheet_name='Monthly Totals')
                            weekly_totals.to_excel(writer, sheet_name='Weekly Totals')
                        st.download_button(
                            label="Download processed data as Excel",
                            data=output.getvalue(),
                            file_name="Analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    elif page == "Fiscal Calendar Totals":
                        st.header("**Fiscal Calendar Totals**")

                        # Calculate fiscal calendar totals using the provided fiscal calendar
                        fiscal_monthly_totals, fiscal_weekly_totals = calculate_fiscal_calendar_totals(df, fiscal_calendar, selected_column)

                        if fiscal_monthly_totals is None or fiscal_weekly_totals is None:
                            st.error("Unable to calculate fiscal totals. Please check your data and fiscal calendar.")
                        else:
                            st.subheader("**Monthly Totals (4-4-5 Fiscal Calendar)**")
                            st.write(fiscal_monthly_totals)

                            st.subheader("**Weekly Totals (4-4-5 Fiscal Calendar)**")
                            st.write(fiscal_weekly_totals)

                            st.header("Export Processed Data")
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                fiscal_monthly_totals.to_excel(writer, sheet_name='Fiscal Monthly Totals')
                                fiscal_weekly_totals.to_excel(writer, sheet_name='Fiscal Weekly Totals')
                            st.download_button(
                                label="Download fiscal calendar totals as Excel",
                                data=output.getvalue(),
                                file_name="Fiscal_Calendar_Totals.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

if __name__ == "__main__":
    main()
