import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
import statsmodels.api as sm
from scipy import stats
import io
import os
from streamlit.components.v1 import html
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import calendar
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv

# Set page config
st.set_page_config(page_title="Advanced Data Analysis App", layout="wide")

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

@st.cache_data
def load_and_preprocess_data(df, selected_column):
    """Load and preprocess the data from an uploaded file."""
    try:
        if df.empty or len(df.columns) <= 1:
            st.error("Unable to parse the CSV file. Please check the file format and try again.")
            return None
        
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
        
        # Convert selected column to numeric, treating blanks as NaN
        df[selected_column] = pd.to_numeric(df[selected_column].replace('', np.nan), errors='coerce')
        df.set_index(date_column, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error in loading and preprocessing data: {str(e)}")
        return None

@st.cache_data
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

@st.cache_data
def extract_fiscal_time_features(df, fiscal_calendar):
    """Extract fiscal time-based features and ensure date alignment."""
    try:
        # Ensure the Date column in both DataFrames is in datetime format
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        fiscal_calendar['Date'] = pd.to_datetime(fiscal_calendar['Date'], errors='coerce')

        # Drop rows with NaT in Date column in either DataFrame
        df.dropna(subset=['Date'], inplace=True)
        fiscal_calendar.dropna(subset=['Date'], inplace=True)

        # Ensure there are no duplicate dates in the fiscal calendar
        fiscal_calendar = fiscal_calendar.drop_duplicates(subset=['Date'])

        # Check if required columns are in the fiscal calendar
        required_columns = ['FiscalYear', 'FiscalMonth', 'FiscalWeek']
        missing_columns = [col for col in required_columns if col not in fiscal_calendar.columns]
        if missing_columns:
            st.error(f"Fiscal calendar is missing required columns: {missing_columns}")
            return None

        # Ensure the date range of the fiscal calendar covers the main data
        min_date, max_date = df['Date'].min(), df['Date'].max()
        if not (fiscal_calendar['Date'].min() <= min_date and fiscal_calendar['Date'].max() >= max_date):
            st.error(f"The date range in the fiscal calendar does not cover the date range in the data. Adjust your fiscal calendar.")
            st.error(f"Data Date Range: {min_date} to {max_date}")
            st.error(f"Fiscal Calendar Date Range: {fiscal_calendar['Date'].min()} to {fiscal_calendar['Date'].max()}")
            return None

        # Merge the fiscal calendar with the data on the Date column
        merged_df = pd.merge(df, fiscal_calendar, how='left', on='Date')

        # Check if the merge was successful and all fiscal columns are present
        missing_cols_after_merge = [col for col in required_columns if col not in merged_df.columns]
        if missing_cols_after_merge:
            st.error(f"Merge failed: Columns {missing_cols_after_merge} are missing after the merge. Please check the fiscal calendar and the date ranges.")
            return None

        if merged_df[required_columns].isnull().any().any():
            st.error("Merge resulted in missing fiscal data. Please ensure the dates match between your data and fiscal calendar.")
            st.write("Possible issues:")
            st.write("- Dates in the data and the fiscal calendar do not match.")
            st.write("- Fiscal calendar has missing values or incorrect column names.")
            return None

        merged_df.set_index('Date', inplace=True)
        return merged_df

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

@st.cache_data
def perform_time_series_analysis(df, selected_column):
    """Perform enhanced time series analysis including decomposition, rolling averages, and outlier detection."""
    decomposition = seasonal_decompose(df[selected_column].dropna(), model='additive', period=30)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid
    df['Outlier'] = np.abs(df['Residual']) > (2 * df['Residual'].std())
    
    # Calculate rolling averages
    df['RollingMean'] = df[selected_column].rolling(window=30).mean()
    df['RollingStd'] = df[selected_column].rolling(window=30).std()

    return df, decomposition

@st.cache_data
def perform_fourier_analysis(df, selected_column):
    """Perform Fourier analysis on the selected column of the time series data."""
    n = len(df)
    freq = np.fft.fftfreq(n)
    mask = freq > 0
    fft_values = np.fft.fft(df[selected_column].fillna(0).values)
    
    # Get power spectrum
    power = np.abs(fft_values) ** 2
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq[mask], y=power[mask], mode='lines', name='Power Spectrum'))
    
    fig.update_layout(
        title="Fourier Analysis (Power Spectrum)",
        xaxis_title="Frequency",
        yaxis_title="Power",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)"  # Transparent background
    )
    
    st.plotly_chart(fig)

@st.cache_data
def perform_stl_decomposition(df, selected_column):
    """Perform STL decomposition on the selected column of the time series data."""
    stl = STL(df[selected_column], seasonal=13)
    result = stl.fit()
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
    fig.add_trace(go.Scatter(x=result.observed.index, y=result.observed, mode='lines', name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residual'), row=4, col=1)
    
    fig.update_layout(height=800, title_text="STL Decomposition", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)

@st.cache_data
def create_monthly_box_plots(df, selected_column, use_fiscal_calendar=False):
    """Create separate interactive box plots for each month of every year."""
    # Ensure the 'Month' column exists
    if not use_fiscal_calendar and 'Month' not in df.columns:
        st.error("The 'Month' column is missing. Ensure that time features are extracted properly.")
        return []

    if use_fiscal_calendar:
        if 'FiscalMonth' not in df.columns:
            st.error("The 'FiscalMonth' column is missing. Ensure that the fiscal calendar is applied correctly.")
            return []
        month_names = df['FiscalMonth'].unique()
        month_names.sort()
        month_label = 'FiscalMonth'
        year_label = 'FiscalYear'
    else:
        month_names = range(1, 13)
        month_label = 'Month'
        year_label = 'Year'
    
    plots = []
    
    for month in month_names:
        monthly_data = df[df[month_label] == month]
        title_suffix = 'Fiscal ' if use_fiscal_calendar else ''
        fig = px.box(monthly_data, x=year_label, y=selected_column, title=f'{title_suffix}{calendar.month_name[int(month)]} Box Plot')
        plots.append(fig)
    return plots

@st.cache_data
def calculate_fiscal_calendar_totals(df, fiscal_calendar, selected_column):
    """Calculate monthly and weekly totals based on the provided fiscal calendar."""
    
    required_columns = ['Date', 'FiscalYear', 'FiscalMonth', 'FiscalWeek']
    if not all(col in fiscal_calendar.columns for col in required_columns):
        st.error(f"Fiscal calendar is missing one or more required columns: {required_columns}")
        return None, None

    df = extract_fiscal_time_features(df, fiscal_calendar)

    if df is None:
        st.error("Failed to extract fiscal time features. Please check your fiscal calendar.")
        return None, None

    fiscal_monthly_totals = df.groupby(['FiscalYear', 'FiscalMonth'])[selected_column].sum()
    fiscal_weekly_totals = df.groupby(['FiscalYear', 'FiscalWeek'])[selected_column].sum()

    fiscal_monthly_totals_df = fiscal_monthly_totals.unstack()
    fiscal_weekly_totals_df = fiscal_weekly_totals.unstack()

    return fiscal_monthly_totals_df, fiscal_weekly_totals_df

def load_fiscal_calendar(uploaded_file):
    """Load the fiscal calendar from an Excel file."""
    try:
        fiscal_calendar = pd.read_excel(uploaded_file)
        fiscal_calendar['Date'] = pd.to_datetime(fiscal_calendar['Date'])

        required_columns = ['FiscalYear', 'FiscalMonth', 'FiscalWeek']
        if not all(col in fiscal_calendar.columns for col in required_columns):
            st.error("Fiscal calendar file must contain columns: FiscalYear, FiscalMonth, FiscalWeek.")
            return None

        return fiscal_calendar
    except Exception as e:
        st.error(f"Error in loading fiscal calendar: {str(e)}")
        return None

def perform_advanced_regression_analysis(df, target_column, independent_columns, method):
    """Perform advanced regression analysis based on selected method."""
    X = pd.get_dummies(df[independent_columns].apply(pd.to_numeric, errors='coerce').dropna(), drop_first=True)
    y = pd.to_numeric(df[target_column], errors='coerce').loc[X.index]
    
    if X.empty or y.empty:
        st.error("The data contains no valid numeric entries after processing. Please check your data.")
        return None, None, None, None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if method == "Linear Regression":
        model = LinearRegression()
    elif method == "Polynomial Regression":
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    elif method == "Ridge Regression":
        model = Ridge()
    elif method == "Lasso Regression":
        model = Lasso()
    elif method == "Elastic Net Regression":
        model = ElasticNet()
    elif method == "Decision Tree Regression":
        model = DecisionTreeRegressor()
    elif method == "Support Vector Regression (SVR)":
        model = SVR()
    elif method == "KNN Regression":
        model = KNeighborsRegressor()
    elif method == "Gaussian Process Regression":
        model = GaussianProcessRegressor()
    elif method == "Neural Network Regression":
        model = MLPRegressor(max_iter=1000)
    else:
        st.error(f"Unknown regression method: {method}")
        return None, None, None, None, None, None
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Detailed results
    st.write(f"**{method} Results**")
    if hasattr(model, 'coef_'):
        st.write(f"Coefficients: {model.coef_}")
    if hasattr(model, 'intercept_'):
        st.write(f"Intercept: {model.intercept_}")
    score = model.score(X_test, y_test)
    st.write(f"R² Score: {score:.4f}")
    
    return model, predictions, X_train, X_test, y_train, y_test

@st.cache_data
def perform_kmeans_clustering(df, target_columns, n_clusters=3, impute_method="drop"):
    """Perform K-means clustering."""
    try:
        # Select only the target columns
        data = df[target_columns]
        
        # Handle missing values based on the selected impute method
        if impute_method == "drop":
            data = data.dropna()
        elif impute_method == "mean":
            data = data.fillna(data.mean())
        elif impute_method == "median":
            data = data.fillna(data.median())
        elif impute_method == "mode":
            data = data.fillna(data.mode().iloc[0])
        else:
            st.error(f"Unknown imputation method: {impute_method}")
            return None
        
        if data.isnull().any().any():
            st.error("There are still missing values after the selected imputation. Please check your data.")
            return None
        
        # One-hot encode categorical columns
        data = pd.get_dummies(data, drop_first=True)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(data)
        
        df['Cluster'] = clusters
        return df
    except Exception as e:
        st.error(f"Error in performing K-means clustering: {str(e)}")
        return None

# New function for outlier detection
def detect_outliers(df, column, method="zscore", threshold=3):
    """Detect outliers using Z-score or IQR method."""
    if method == "zscore":
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = df[z_scores > threshold]
    elif method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    return outliers

def handle_missing_data(df, column, method="mean"):
    """Handle missing data using the specified method and return the modified DataFrame along with an interactive plot."""
    original_df = df[[column]].copy()  # Keep a copy of the original data for comparison
    
    # Handle missing data
    if method == "mean":
        df[column] = df[column].fillna(df[column].mean())
    elif method == "median":
        df[column] = df[column].fillna(df[column].median())
    elif method == "mode":
        df[column] = df[column].fillna(df[column].mode().iloc[0])
    elif method == "ffill":
        df[column] = df[column].fillna(method='ffill')
    elif method == "bfill":
        df[column] = df[column].fillna(method='bfill')
    elif method == "interpolate":
        df[column] = df[column].interpolate()

    # Create an interactive plot comparing the original and handled time series
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=original_df.index, y=original_df[column], mode='lines', name='Original Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=f'{method.capitalize()} Imputation', line=dict(color='orange')))
    
    fig.update_layout(title=f'Missing Data Handling: {method.capitalize()} Method',
                      xaxis_title='Date',
                      yaxis_title=column,
                      plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                      paper_bgcolor="rgba(0,0,0,0)"  # Transparent background
    )
    
    st.plotly_chart(fig)
    
    # Display the modified dataframe
    st.write("Data after handling missing values:")
    st.write(df)
    
    # Create a comparison dataframe
    comparison_df = pd.DataFrame({
        "Original": original_df[column],
        "After Handling": df[column]
    })

    st.write("Comparison of Original and Handled Data:")
    st.write(comparison_df)
    
    # Create an interactive plot comparing the original and handled data together
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=original_df.index, y=original_df[column], mode='lines', name='Original Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=f'Handled Data', line=dict(color='orange')))
    
    fig.update_layout(title='Comparison of Original and Handled Data',
                      xaxis_title='Date',
                      yaxis_title=column,
                      plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                      paper_bgcolor="rgba(0,0,0,0)"  # Transparent background
    )
    
    st.plotly_chart(fig)

    return df

# Outlier detection with visualization and Bell Curve addition
def visualize_outliers(df, column, method="zscore", threshold=3, action="remove"):
    """Visualize and handle outliers, and return the modified DataFrame."""
    outliers = detect_outliers(df, column, method, threshold)
    
    # Filtering options
    st.subheader("Filter options for Bell Curve Plot")
    filter_options = st.multiselect("Filter by:", ["Year", "Quarter", "Month", "WeekOfYear", "WeekOfMonth", "DayOfYear", "DayOfMonth", "DayOfWeekName"])

    filtered_df = df.copy()

    for filter_option in filter_options:
        if filter_option == "Year":
            selected_years = st.multiselect("Select year(s):", df['Year'].unique())
            if selected_years:
                filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
        elif filter_option == "Quarter":
            selected_quarters = st.multiselect("Select quarter(s):", df['Quarter'].unique())
            if selected_quarters:
                filtered_df = filtered_df[filtered_df['Quarter'].isin(selected_quarters)]
        elif filter_option == "Month":
            selected_months = st.multiselect("Select month(s):", df['Month'].unique())
            if selected_months:
                filtered_df = filtered_df[filtered_df['Month'].isin(selected_months)]
        elif filter_option == "WeekOfYear":
            selected_weeks = st.multiselect("Select week(s) of the year:", df['WeekOfYear'].unique())
            if selected_weeks:
                filtered_df = filtered_df[filtered_df['WeekOfYear'].isin(selected_weeks)]
        elif filter_option == "WeekOfMonth":
            selected_weeks_of_month = st.multiselect("Select week(s) of the month:", df['WeekOfMonth'].unique())
            if selected_weeks_of_month:
                filtered_df = filtered_df[filtered_df['WeekOfMonth'].isin(selected_weeks_of_month)]
        elif filter_option == "DayOfYear":
            selected_days_of_year = st.multiselect("Select day(s) of the year:", df['DayOfYear'].unique())
            if selected_days_of_year:
                filtered_df = filtered_df[filtered_df['DayOfYear'].isin(selected_days_of_year)]
        elif filter_option == "DayOfMonth":
            selected_days_of_month = st.multiselect("Select day(s) of the month:", df['DayOfMonth'].unique())
            if selected_days_of_month:
                filtered_df = filtered_df[filtered_df['DayOfMonth'].isin(selected_days_of_month)]
        elif filter_option == "DayOfWeekName":
            selected_days_of_week = st.multiselect("Select day(s) of the week:", df['DayOfWeekName'].unique())
            if selected_days_of_week:
                filtered_df = filtered_df[filtered_df['DayOfWeekName'].isin(selected_days_of_week)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df[column], mode='lines', name='Original Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=outliers.index, y=outliers[column], mode='markers', name='Outliers', marker=dict(color='red')))
    
    fig.update_layout(title=f'Outliers Detection: {method.capitalize()} Method',
                      xaxis_title='Date',
                      yaxis_title=column,
                      plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                      paper_bgcolor="rgba(0,0,0,0)"  # Transparent background
    )
    
    st.plotly_chart(fig)
    
    if action == "remove":
        filtered_df = filtered_df[~filtered_df.index.isin(outliers.index)]
        st.write("Outliers removed from the dataset.")
    elif action == "cap":
        cap_value = st.slider("Select Cap Value", min_value=float(outliers[column].min()), max_value=float(outliers[column].max()))
        filtered_df[column] = np.where(filtered_df[column] > cap_value, cap_value, filtered_df[column])
        st.write("Outliers capped in the dataset.")
    
    # Display the modified dataframe
    st.write("Data after handling outliers:")
    st.write(filtered_df)
    
    # Bell Curve Plot
    st.subheader("Bell Curve Plot (1, 2, 3 Standard Deviations)")

    mean_value = filtered_df[column].mean()
    std_dev = filtered_df[column].std()
    x = np.linspace(mean_value - 4 * std_dev, mean_value + 4 * std_dev, 1000)
    y = stats.norm.pdf(x, mean_value, std_dev)

    fig_bell_curve = go.Figure()
    fig_bell_curve.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Distribution', line=dict(color='blue')))
    
    # Add mean and standard deviation lines with annotations
    fig_bell_curve.add_trace(go.Scatter(x=[mean_value, mean_value], y=[0, max(y)], mode='lines', name='Mean', line=dict(color='green', dash='dash')))
    fig_bell_curve.add_annotation(x=mean_value, y=max(y), text=f"Mean: {mean_value:.2f}", showarrow=True, arrowhead=2, ax=40, ay=-40)

    fig_bell_curve.add_trace(go.Scatter(x=[mean_value + std_dev, mean_value + std_dev], y=[0, max(y)], mode='lines', name='+1 Std Dev', line=dict(color='orange', dash='dot')))
    fig_bell_curve.add_annotation(x=mean_value + std_dev, y=max(y) * 0.9, text=f"+1 Std Dev: {mean_value + std_dev:.2f}", showarrow=True, arrowhead=2, ax=40, ay=-40)

    fig_bell_curve.add_trace(go.Scatter(x=[mean_value - std_dev, mean_value - std_dev], y=[0, max(y)], mode='lines', name='-1 Std Dev', line=dict(color='orange', dash='dot')))
    fig_bell_curve.add_annotation(x=mean_value - std_dev, y=max(y) * 0.9, text=f"-1 Std Dev: {mean_value - std_dev:.2f}", showarrow=True, arrowhead=2, ax=-40, ay=-40)

    fig_bell_curve.add_trace(go.Scatter(x=[mean_value + 2 * std_dev, mean_value + 2 * std_dev], y=[0, max(y)], mode='lines', name='+2 Std Dev', line=dict(color='red', dash='dot')))
    fig_bell_curve.add_annotation(x=mean_value + 2 * std_dev, y=max(y) * 0.7, text=f"+2 Std Dev: {mean_value + 2 * std_dev:.2f}", showarrow=True, arrowhead=2, ax=40, ay=-40)

    fig_bell_curve.add_trace(go.Scatter(x=[mean_value - 2 * std_dev, mean_value - 2 * std_dev], y=[0, max(y)], mode='lines', name='-2 Std Dev', line=dict(color='red', dash='dot')))
    fig_bell_curve.add_annotation(x=mean_value - 2 * std_dev, y=max(y) * 0.7, text=f"-2 Std Dev: {mean_value - 2 * std_dev:.2f}", showarrow=True, arrowhead=2, ax=-40, ay=-40)

    fig_bell_curve.add_trace(go.Scatter(x=[mean_value + 3 * std_dev, mean_value + 3 * std_dev], y=[0, max(y)], mode='lines', name='+3 Std Dev', line=dict(color='purple', dash='dot')))
    fig_bell_curve.add_annotation(x=mean_value + 3 * std_dev, y=max(y) * 0.5, text=f"+3 Std Dev: {mean_value + 3 * std_dev:.2f}", showarrow=True, arrowhead=2, ax=40, ay=-40)

    fig_bell_curve.add_trace(go.Scatter(x=[mean_value - 3 * std_dev, mean_value - 3 * std_dev], y=[0, max(y)], mode='lines', name='-3 Std Dev', line=dict(color='purple', dash='dot')))
    fig_bell_curve.add_annotation(x=mean_value - 3 * std_dev, y=max(y) * 0.5, text=f"-3 Std Dev: {mean_value - 3 * std_dev:.2f}", showarrow=True, arrowhead=2, ax=-40, ay=-40)

    fig_bell_curve.update_layout(
        title="Bell Curve with Standard Deviations",
        xaxis_title="Value",
        yaxis_title="Probability Density",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified"
    )

    st.plotly_chart(fig_bell_curve)

    return filtered_df
# Missing Data Visualization with Plotly
def visualize_missing_data(df):
    """Visualize missing data using an interactive Plotly heatmap."""
    st.subheader("Missing Data Visualization")
    st.write("The following heatmap shows the missing data across the dataset:")

    missing_df = df.isnull().astype(int)
    fig = px.imshow(missing_df, aspect="auto", color_continuous_scale='viridis')
    fig.update_layout(
        title="Missing Data Heatmap",
        xaxis_title="Columns",
        yaxis_title="Rows",
        coloraxis_showscale=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig)

# Correlation Heatmap
def correlation_heatmap(df):
    """Display a correlation heatmap of the dataframe."""
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    st.pyplot(plt)

def enhanced_monthly_totals_plot(monthly_totals_pivot, selected_column):
    """Create an enhanced interactive plot for monthly totals with no background."""
    fig = go.Figure()

    # Check if the columns are already month names or need to be converted
    if isinstance(monthly_totals_pivot.columns[0], int):
        months = [calendar.month_name[month] for month in monthly_totals_pivot.columns]
    else:
        months = monthly_totals_pivot.columns

    for year in monthly_totals_pivot.index:
        fig.add_trace(
            go.Bar(
                x=months,
                y=monthly_totals_pivot.loc[year],
                name=str(year),
                text=[f"{month}: {value}" for month, value in zip(months, monthly_totals_pivot.loc[year])],
                hoverinfo="text",
                hoverlabel=dict(namelength=-1),
            )
        )

    fig.update_layout(
        title="Monthly Totals (Interactive Plot)",
        xaxis_title="Month",
        yaxis_title=f"Total {selected_column}",
        barmode='group',
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        legend_title="Year",
        xaxis=dict(type='category'),
        yaxis=dict(showgrid=True, gridcolor='LightGray'),
        margin=dict(l=0, r=0, t=50, b=50),
    )

    st.plotly_chart(fig)


def enhanced_weekly_totals_plot(weekly_totals_pivot, selected_column):
    """Create an enhanced interactive plot for weekly totals with no background."""
    fig = go.Figure()

    for year in weekly_totals_pivot.index:
        fig.add_trace(
            go.Scatter(
                x=[f"Week {week}" for week in weekly_totals_pivot.columns],
                y=weekly_totals_pivot.loc[year],
                mode='lines+markers',
                name=str(year),
                text=[f"Week {week}: {value}" for week, value in zip(weekly_totals_pivot.columns, weekly_totals_pivot.loc[year])],
                hoverinfo="text",
                hoverlabel=dict(namelength=-1),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        title="Weekly Totals (Interactive Plot)",
        xaxis_title="Week",
        yaxis_title=f"Total {selected_column}",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        legend_title="Year",
        xaxis=dict(type='category'),
        yaxis=dict(showgrid=True, gridcolor='LightGray'),
        margin=dict(l=0, r=0, t=50, b=50),
    )

    st.plotly_chart(fig)


def enhanced_fiscal_monthly_totals_plot(fiscal_monthly_totals, selected_column):
    """Create an enhanced interactive plot for fiscal monthly totals with no background."""
    fig = go.Figure()

    for year in fiscal_monthly_totals.index:
        fig.add_trace(
            go.Bar(
                x=[f"Fiscal Month {int(month)}" for month in fiscal_monthly_totals.columns],
                y=fiscal_monthly_totals.loc[year],
                name=str(year),
                text=[f"Fiscal Month {int(month)}: {value}" for month, value in zip(fiscal_monthly_totals.columns, fiscal_monthly_totals.loc[year])],
                hoverinfo="text",
                hoverlabel=dict(namelength=-1),
            )
        )

    fig.update_layout(
        title="Fiscal Monthly Totals (Interactive Plot)",
        xaxis_title="Fiscal Month",
        yaxis_title=f"Total {selected_column}",
        barmode='group',
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        legend_title="Fiscal Year",
        xaxis=dict(type='category'),
        yaxis=dict(showgrid=True, gridcolor='LightGray'),
        margin=dict(l=0, r=0, t=50, b=50),
    )

    st.plotly_chart(fig)


def enhanced_fiscal_weekly_totals_plot(fiscal_weekly_totals, selected_column):
    """Create an enhanced interactive plot for fiscal weekly totals with no background."""
    fig = go.Figure()

    for year in fiscal_weekly_totals.index:
        fig.add_trace(
            go.Scatter(
                x=[f"Fiscal Week {int(week)}" for week in fiscal_weekly_totals.columns],
                y=fiscal_weekly_totals.loc[year],
                mode='lines+markers',
                name=str(year),
                text=[f"Fiscal Week {int(week)}: {value}" for week, value in zip(fiscal_weekly_totals.columns, fiscal_weekly_totals.loc[year])],
                hoverinfo="text",
                hoverlabel=dict(namelength=-1),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        title="Fiscal Weekly Totals (Interactive Plot)",
        xaxis_title="Fiscal Week",
        yaxis_title=f"Total {selected_column}",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        legend_title="Fiscal Year",
        xaxis=dict(type='category'),
        yaxis=dict(showgrid=True, gridcolor='LightGray'),
        margin=dict(l=0, r=0, t=50, b=50),
    )

    st.plotly_chart(fig)

# Standard seasonality analysis function with multi-select filters
def standard_seasonality_analysis_page(df, selected_column, calendar_type):
    st.header("**Seasonality Analysis**")

    filter_options = st.multiselect("Filter by:", ["Year", "Quarter", "Month", "WeekOfYear", "WeekOfMonth", "DayOfYear", "DayOfMonth", "DayOfWeekName"])
    
    filtered_df = df.copy()

    for filter_option in filter_options:
        if filter_option == "Year":
            selected_years = st.multiselect("Select year(s):", df['Year'].unique())
            if selected_years:
                filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
        elif filter_option == "Quarter":
            selected_quarters = st.multiselect("Select quarter(s):", df['Quarter'].unique())
            if selected_quarters:
                filtered_df = filtered_df[filtered_df['Quarter'].isin(selected_quarters)]
        elif filter_option == "Month":
            selected_months = st.multiselect("Select month(s):", df['Month'].unique())
            if selected_months:
                filtered_df = filtered_df[filtered_df['Month'].isin(selected_months)]
        elif filter_option == "WeekOfYear":
            selected_weeks = st.multiselect("Select week(s) of the year:", df['WeekOfYear'].unique())
            if selected_weeks:
                filtered_df = filtered_df[filtered_df['WeekOfYear'].isin(selected_weeks)]
        elif filter_option == "WeekOfMonth":
            selected_weeks_of_month = st.multiselect("Select week(s) of the month:", df['WeekOfMonth'].unique())
            if selected_weeks_of_month:
                filtered_df = filtered_df[filtered_df['WeekOfMonth'].isin(selected_weeks_of_month)]
        elif filter_option == "DayOfYear":
            selected_days_of_year = st.multiselect("Select day(s) of the year:", df['DayOfYear'].unique())
            if selected_days_of_year:
                filtered_df = filtered_df[filtered_df['DayOfYear'].isin(selected_days_of_year)]
        elif filter_option == "DayOfMonth":
            selected_days_of_month = st.multiselect("Select day(s) of the month:", df['DayOfMonth'].unique())
            if selected_days_of_month:
                filtered_df = filtered_df[filtered_df['DayOfMonth'].isin(selected_days_of_month)]
        elif filter_option == "DayOfWeekName":
            selected_days_of_week = st.multiselect("Select day(s) of the week:", df['DayOfWeekName'].unique())
            if selected_days_of_week:
                filtered_df = filtered_df[filtered_df['DayOfWeekName'].isin(selected_days_of_week)]

    # Seasonality fields to plot
    seasonality_fields = ['Year', 'Quarter', 'Month', 'WeekOfYear', 'WeekOfMonth', 'DayOfYear', 'DayOfMonth', 'DayOfWeekName']
    
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

def perform_rolling_statistics(df, selected_column):
    """Perform rolling statistics with different window sizes."""
    window_size = st.slider("Select rolling window size (days)", 7, 365, 30)
    
    df[f'RollingMean_{window_size}'] = df[selected_column].rolling(window=window_size).mean()
    df[f'RollingMedian_{window_size}'] = df[selected_column].rolling(window=window_size).median()
    df[f'RollingStd_{window_size}'] = df[selected_column].rolling(window=window_size).std()
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Rolling Mean", "Rolling Median", "Rolling Standard Deviation"))
    fig.add_trace(go.Scatter(x=df.index, y=df[f'RollingMean_{window_size}'], mode='lines', name=f'Rolling Mean ({window_size} days)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[f'RollingMedian_{window_size}'], mode='lines', name=f'Rolling Median ({window_size} days)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[f'RollingStd_{window_size}'], mode='lines', name=f'Rolling Std Dev ({window_size} days)'), row=3, col=1)
    
    fig.update_layout(height=800, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)

def interactive_data_transformation(df):
    """Provide an interactive interface for data transformation and cleaning."""
    st.header("**Interactive Data Transformation and Cleaning**")
    
    transformation_type = st.selectbox("Select Transformation Type", ["Standardization", "Normalization", "Log Transformation", "Square Root Transformation", "Box-Cox Transformation"])
    
    column_to_transform = st.selectbox("Select Column to Transform", df.select_dtypes(include=[np.number]).columns.tolist())
    
    if transformation_type == "Standardization":
        scaler = StandardScaler()
        df[column_to_transform] = scaler.fit_transform(df[[column_to_transform]])
    elif transformation_type == "Normalization":
        scaler = MinMaxScaler()
        df[column_to_transform] = scaler.fit_transform(df[[column_to_transform]])
    elif transformation_type == "Log Transformation":
        df[column_to_transform] = np.log1p(df[column_to_transform])
    elif transformation_type == "Square Root Transformation":
        df[column_to_transform] = np.sqrt(df[column_to_transform])
    elif transformation_type == "Box-Cox Transformation":
        df[column_to_transform], _ = stats.boxcox(df[column_to_transform].clip(lower=0.0001))  # Avoid zero or negative values
    
    st.write(f"Data after {transformation_type}:")
    st.write(df.head())

    fig = px.histogram(df, x=column_to_transform, nbins=30, title=f"Histogram of {column_to_transform} after {transformation_type}")
    st.plotly_chart(fig)

def automated_eda(df):
    """Run automated EDA using Sweetviz."""
    eda_tool = st.selectbox("Select EDA Tool", ["Sweetviz"])
    
    if eda_tool == "Sweetviz":
        report = sv.analyze(df)
        report.show_html(filepath="sweetviz_report.html", open_browser=False)
        st.markdown("### Sweetviz EDA Report")
        with open("sweetviz_report.html", "r", encoding="utf-8") as f:
            report_html = f.read()
        html(report_html, width=1200, height=800, scrolling=True)

def main():
    st.title("Advanced Data Analysis App")

    st.sidebar.markdown("### Made by Ashwin Nair")

    with st.sidebar.expander("ℹ️ How to Use the App", expanded=False):
        st.markdown(
            """
            **Instructions**  
            - **Upload Data File**: Upload your CSV or Excel file containing the data you want to analyze. 
              Ensure the file has at least two columns:  
              - **Date Column**: The first column should be a date column in one of the following formats: 
                `%d-%b-%y`, `%Y-%m-%d`, `%d/%m/%Y`, or `%m/%d/%Y`.
              - **Value Column**: At least one column should contain numeric data for analysis.
            - **Optional Fiscal Calendar**: Upload a fiscal calendar in Excel format if you wish to perform fiscal year analysis. 
              The fiscal calendar file must contain the following columns:
              - `Date`: Dates in a recognized format.
              - `FiscalYear`: Integer representing the fiscal year.
              - `FiscalMonth`: Integer representing the fiscal month.
              - `FiscalWeek`: Integer representing the fiscal week.
            - **Navigation**: Use the sidebar to select different pages for profiling, data exploration, and time series analysis.
            - **Calendar Selection**: Choose between a normal calendar or fiscal calendar for the analysis if a fiscal calendar is provided.

            **Data Format Example**:  
            ```
            | Date       | Sales |
            |------------|-------|
            | 01-Jan-23  | 200   |
            | 02-Jan-23  | 220   |
            | 03-Jan-23  | 215   |
            ```

            **Fiscal Calendar Example**:  
            ```
            | Date       | FiscalYear | FiscalMonth | FiscalWeek |
            |------------|------------|-------------|------------|
            | 01-Jan-23  | 2023       | 1           | 1          |
            | 02-Jan-23  | 2023       | 1           | 1          |
            | 03-Jan-23  | 2023       | 1           | 1          |
            ```
            """
        )

    st.sidebar.title("Upload your CSV or Excel file")
    uploaded_file = st.sidebar.file_uploader("Drag and drop your data file here", type=["csv", "xlsx"])

    st.sidebar.title("Upload Fiscal Calendar")
    fiscal_calendar_file = st.sidebar.file_uploader("Upload your fiscal calendar file here (optional)", type=["xlsx"])

    st.sidebar.title("Calendar Selection")
    calendar_type = st.sidebar.radio("Select Calendar Type:", ("Normal Calendar", "Fiscal Calendar"))

    st.sidebar.title("Navigation")
    st.sidebar.markdown("### Choose a Page", unsafe_allow_html=True)
    pages = ["Data Exploration", "Time Series Analysis", "Seasonality Analysis", "Monthly Box Plots", "Totals Analysis", "Fiscal Calendar Totals", "Regression Analysis", "Clustering", "Profiling", "Outlier Detection", "Missing Data Handling", "Rolling Statistics", "Interactive Data Transformation", "Automated EDA"]
    page = st.sidebar.radio("", pages, key='pages')

    if uploaded_file is not None:
        data_file_path = save_uploaded_file(uploaded_file)
        df = load_data(data_file_path)
        
        fiscal_calendar = None
        if fiscal_calendar_file is not None:
            fiscal_calendar_file_path = save_uploaded_file(fiscal_calendar_file)
            fiscal_calendar = load_fiscal_calendar(fiscal_calendar_file_path)

        if df is not None:
            columns = df.columns.tolist()
            date_column = columns.pop(0)
            
            selected_column = st.selectbox("Select the data column to analyze", columns)
            
            df = load_and_preprocess_data(df, selected_column)

            if df is None:
                st.error("Data could not be loaded or processed. Please check your file and try again.")
                return

            if calendar_type == "Normal Calendar":
                df = extract_time_features(df)
            elif calendar_type == "Fiscal Calendar" and fiscal_calendar is not None:
                df = extract_fiscal_time_features(df, fiscal_calendar)

            if df is None:
                st.error("Failed to process the data with the selected calendar. Please check the fiscal calendar or try using the normal calendar.")
                return

            if page == "Data Exploration":
                st.header("**Data Exploration**")
                st.write(df.describe())

            elif page == "Time Series Analysis":
                st.header("**Enhanced Time Series Analysis**")
                st.subheader("**Time Series of Selected Data**")
                df, decomposition = perform_time_series_analysis(df, selected_column)
                
                if df is None:
                    st.error("Time series analysis failed. Please check the data and try again.")
                    return

                mean_value = df[selected_column].mean()
                std_dev = df[selected_column].std()
                ucl1 = mean_value + 1 * std_dev
                lcl1 = mean_value - 1 * std_dev
                ucl2 = mean_value + 2 * std_dev
                lcl2 = mean_value - 2 * std_dev
                ucl3 = mean_value + 3 * std_dev
                lcl3 = mean_value - 3 * std_dev

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

                st.subheader("**Time Series Decomposition**")
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
                fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'), row=1, col=1)
                fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
                fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
                fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)
                fig.update_layout(height=800, title_text="Time Series Decomposition")
                st.plotly_chart(fig)

                st.subheader("**Rolling Mean and Standard Deviation**")
                fig = px.line(df, x=df.index, y=[df[selected_column], df['RollingMean'], df['RollingStd']], labels={'value': selected_column})
                st.plotly_chart(fig)

            elif page == "Seasonality Analysis":
                st.header("**Advanced Seasonality Analysis**")
                analysis_type = st.selectbox("Select Analysis Type", ["Fourier Analysis", "STL Decomposition", "Standard Analysis"])
                
                if analysis_type == "Fourier Analysis":
                    perform_fourier_analysis(df, selected_column)
                elif analysis_type == "STL Decomposition":
                    perform_stl_decomposition(df, selected_column)
                else:
                    standard_seasonality_analysis_page(df, selected_column, calendar_type)

            elif page == "Monthly Box Plots":
                st.header("**Monthly Box Plots**")
                use_fiscal_calendar = calendar_type == "Fiscal Calendar"
                box_plot_figs = create_monthly_box_plots(df, selected_column, use_fiscal_calendar)
                for box_plot_fig in box_plot_figs:
                    st.plotly_chart(box_plot_fig)

            elif page == "Totals Analysis":
                st.header("**Monthly Totals**")
                numeric_columns = df.select_dtypes(include=[np.number]).columns

                monthly_totals = df.resample('M').sum(numeric_only=True)
                st.write("Monthly Totals (Tabular Format):")
                st.write(monthly_totals[[selected_column]])

                monthly_totals_pivot = monthly_totals.pivot_table(index=monthly_totals.index.year, columns=monthly_totals.index.month, values=selected_column, aggfunc='sum')
                monthly_totals_pivot.columns = [calendar.month_name[i] for i in monthly_totals_pivot.columns]
                st.write("Monthly Totals (Pivot Table Format):")
                st.write(monthly_totals_pivot)

                st.subheader("**Monthly Totals (Interactive Plot)**")
                enhanced_monthly_totals_plot(monthly_totals_pivot, selected_column)

                st.header("**Weekly Totals**")

                weekly_totals = df.resample('W').sum(numeric_only=True)
                weekly_totals['WeekStart'] = weekly_totals.index - pd.to_timedelta(weekly_totals.index.weekday, unit='d')
                weekly_totals.set_index('WeekStart', inplace=True)
                st.write("Weekly Totals (Tabular Format):")
                st.write(weekly_totals[[selected_column]])

                weekly_totals_pivot = weekly_totals.pivot_table(index=weekly_totals.index.year, columns=weekly_totals.index.isocalendar().week, values=selected_column, aggfunc='sum')
                st.write("Weekly Totals (Pivot Table Format):")
                st.write(weekly_totals_pivot)

                st.subheader("**Weekly Totals (Interactive Plot)**")
                enhanced_weekly_totals_plot(weekly_totals_pivot, selected_column)

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

                if fiscal_calendar is not None:
                    fiscal_monthly_totals, fiscal_weekly_totals = calculate_fiscal_calendar_totals(df, fiscal_calendar, selected_column)

                    if fiscal_monthly_totals is None or fiscal_weekly_totals is None:
                        st.error("Unable to calculate fiscal totals. Please check your data and fiscal calendar.")
                    else:
                        st.subheader("**Monthly Totals (4-4-5 Fiscal Calendar)**")
                        st.write(fiscal_monthly_totals)

                        st.subheader("**Weekly Totals (4-4-5 Fiscal Calendar)**")
                        st.write(fiscal_weekly_totals)

                        st.subheader("**Monthly Totals (Interactive Plot)**")
                        enhanced_fiscal_monthly_totals_plot(fiscal_monthly_totals, selected_column)

                        st.subheader("**Weekly Totals (Interactive Plot)**")
                        enhanced_fiscal_weekly_totals_plot(fiscal_weekly_totals, selected_column)

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

            elif page == "Regression Analysis":
                st.header("**Advanced Regression Analysis**")
                
                # Step 1: Select independent and dependent variables
                independent_columns = st.multiselect("Select independent variable(s)", df.columns.tolist(), default=columns)
                target_column = st.selectbox("Select the target variable (dependent variable)", df.columns.tolist())
                
                # Step 2: Choose the regression method
                regression_method = st.selectbox("Select Regression Method", [
                    "Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression", 
                    "Elastic Net Regression", "Decision Tree Regression", "Support Vector Regression (SVR)", 
                    "KNN Regression", "Gaussian Process Regression", "Neural Network Regression"
                ])
                
                # Step 3: Train and visualize the model
                model, predictions, X_train, X_test, y_train, y_test = perform_advanced_regression_analysis(df, target_column, independent_columns, regression_method)
                
                if model is not None and predictions is not None:
                    if len(independent_columns) == 1:
                        fig = px.scatter(df, x=independent_columns[0], y=target_column, title=f"{regression_method}")
                        fig.add_trace(go.Scatter(x=X_train[independent_columns[0]], y=model.predict(X_train), mode='markers', name='Train Predictions', marker=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=X_test[independent_columns[0]], y=predictions, mode='markers', name='Test Predictions', marker=dict(color='red')))
                    else:
                        fig = px.scatter_matrix(pd.DataFrame(X_train, columns=independent_columns), dimensions=independent_columns, color=pd.DataFrame(y_train, columns=[target_column]).values.flatten(), title=f"{regression_method}")
                        fig.add_trace(go.Scatter(x=X_train[independent_columns[0]], y=model.predict(X_train), mode='markers', name='Train Predictions', marker=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=X_test[independent_columns[0]], y=predictions, mode='markers', name='Test Predictions', marker=dict(color='red')))
                    st.plotly_chart(fig)

            elif page == "Clustering":
                        st.header("**Interactive Scatter Plot**")
                        
                        # Title with emoji
                        st.title('✨ Cross-Filterable Scatter Plot from your Data')
                        
                        # Instructions in an expander
                        with st.expander("ℹ️ Instructions"):
                            st.write("""
                                Select the X and Y axes, and optionally choose other attributes such as color and size to customize your scatter plot.
                                You can also filter data points using the lasso or box selection tools directly on the plot.
                            """)
                        
                        # Ensure at least two columns exist
                        if len(df.columns) < 2:
                            st.error('The data must have at least two columns.')
                            st.stop()
                        
                        # Use a layout with columns for axis selection and advanced options
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            x_axis = st.selectbox('Select X-axis', options=df.columns)
                        with col2:
                            y_axis = st.selectbox('Select Y-axis', options=df.columns, index=1)
                        with col3:
                            color_option = st.selectbox('Select Color', options=[None] + df.columns.tolist(), index=0)
                        
                        # Advanced customization options placed on the main page now
                        col4, col5 = st.columns([1, 1])
                        
                        with col4:
                            size_option = st.selectbox('Select Size', options=[None] + df.columns.tolist(), index=0)
                        with col5:
                            hover_data_options = st.multiselect('Select Hover Data', options=df.columns)
                        
                        # Create scatter plot with Plotly
                        fig = px.scatter(
                            df,
                            x=x_axis,
                            y=y_axis,
                            color=color_option if color_option else None,
                            size=size_option if size_option else None,
                            hover_data=hover_data_options,
                            title=f'Scatter Plot of {y_axis} vs {x_axis}',
                            template='plotly_dark',  # Set Plotly theme
                            labels={x_axis: f"{x_axis} (units)", y_axis: f"{y_axis} (units)"}
                        )
                        
                        # Improve plot layout
                        fig.update_layout(
                            margin=dict(l=20, r=20, t=50, b=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(size=14),
                        )
                        
                        # Add lasso/box selection tool and capture selection using streamlit-plotly-events
                        from streamlit_plotly_events import plotly_events
                        
                        st.write("Use the lasso or box tool in the plot to select data points.")
                        selected_points = plotly_events(
                            fig,
                            click_event=False,
                            hover_event=False,
                            select_event=True,
                            override_height=600,
                            override_width='100%'
                        )
                        
                        # Check if selection exists
                        if selected_points:
                            # selected_points is a list of dictionaries with 'curveNumber', 'pointNumber', etc.
                            selected_indices = [pt['pointIndex'] for pt in selected_points]
                        
                            # Filter dataframe based on selection
                            filtered_df = df.iloc[selected_indices]
                        
                            st.write('### Filtered Data')
                            st.dataframe(filtered_df)
                        
                            # Optionally, create another plot for filtered data
                            st.write('### Scatter Plot of Filtered Data')
                            filtered_fig = px.scatter(
                                filtered_df,
                                x=x_axis,
                                y=y_axis,
                                color=color_option if color_option else None,
                                size=size_option if size_option else None,
                                hover_data=hover_data_options,
                                title=f'Filtered Scatter Plot of {y_axis} vs {x_axis}'
                            )
                            st.plotly_chart(filtered_fig, use_container_width=True)
                        
                            # Option to download the filtered data
                            st.download_button(
                                label="Download filtered data as CSV",
                                data=filtered_df.to_csv().encode('utf-8'),
                                file_name="filtered_data.csv",
                                mime='text/csv'
                            )
                        else:
                            st.info('Select points using the lasso or box tool to filter data.')











            elif page == "Profiling":
                st.title("Profiling App by Ashwin")
                if df is not None:
                    profile = ProfileReport(df, title="Profiling Report")
                    profile.to_file("report.html")
                    
                    with open("report.html", "r", encoding='utf-8') as f:
                        report_html = f.read()
                    
                    html(report_html, width=1200, height=800, scrolling=True)

            elif page == "Outlier Detection":
                st.header("**Outlier Detection**")
                method = st.selectbox("Select Outlier Detection Method", ["zscore", "iqr"])
                threshold = st.slider("Select Threshold (for Z-score)", 1.0, 4.0, 3.0)
                
                action = st.radio("Select Action for Outliers", ["remove", "cap"])
                df = visualize_outliers(df, selected_column, method, threshold, action)

            elif page == "Missing Data Handling":
                st.header("**Interactive Missing Data Handling**")
                visualize_missing_data(df)
                
                method = st.selectbox("Select Missing Data Handling Method", ["mean", "median", "mode", "ffill", "bfill", "interpolate"])
                df = handle_missing_data(df, selected_column, method)
                st.write(f"Missing data handled using {method} method.")
                st.write(df.describe())

            elif page == "Rolling Statistics":
                st.header("**Rolling Statistics**")
                perform_rolling_statistics(df, selected_column)

            elif page == "Interactive Data Transformation":
                st.header("**Interactive Data Transformation**")
                interactive_data_transformation(df)

            elif page == "Automated EDA":
                st.header("**Automated Exploratory Data Analysis (EDA)**")
                automated_eda(df)

    else:
        st.info("Please upload a data file to proceed.")

if __name__ == "__main__":
    main()
