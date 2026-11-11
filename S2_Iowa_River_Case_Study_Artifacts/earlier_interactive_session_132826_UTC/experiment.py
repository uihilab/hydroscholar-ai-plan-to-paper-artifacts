# HydroScholar Experiment Script: experiment.py
# -*- coding: utf-8 -*-

# --- Standard Imports ---
import pandas as pd
import numpy as np
# Use Agg backend for non-interactive plotting in Streamlit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import os
import json
import sys
from scipy import stats
from io import StringIO
import datetime

# --- Optional Imports (Uncomment as needed) ---
# import pymannkendall as mk
# from scipy.optimize import curve_fit
# from scipy.stats import pearson3, skew, norm

# --- Environment Setup (Example) ---
# Load API keys or configurations if necessary
# noaa_token = os.environ.get('NOAA_TOKEN')
# openai_api_key = os.environ.get('OPENAI_API_KEY')

def fetch_streamflow_data(usgs_site_number: str, start_date_str: str, end_date_str: str):
    """
    Fetch streamflow data from USGS water services with extensive error handling.

    Parameters:
        usgs_site_number (str): USGS site number.
        start_date_str (str): Start date in YYYY-MM-DD format.
        end_date_str (str): End date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: DataFrame with columns 'date' and 'discharge' if successful, else None.
    """
    base_url = "https://waterservices.usgs.gov/nwis/dv/"
    params = {
        "format": "json",
        "sites": usgs_site_number,
        "startDT": start_date_str,
        "endDT": end_date_str,
        "parameterCd": "00060",
        "siteStatus": "all"
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from USGS: {e}")
        return None

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return None

    try:
        time_series = data["value"]["timeSeries"][0]["values"][0]["value"]
    except (KeyError, IndexError) as e:
        print(f"Error extracting time series data from response: {e}")
        return None

    records = []
    for entry in time_series:
        try:
            date = pd.to_datetime(entry.get("dateTime"))
            discharge = float(entry.get("value"))
            records.append({"date": date, "discharge": discharge})
        except (ValueError, TypeError) as e:
            print(f"Error processing entry {entry}: {e}")
    if not records:
        print("No valid data records found.")
        return None

    try:
        df = pd.DataFrame(records)
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return None

    # Validate that required columns exist
    required_columns = {"date", "discharge"}
    if not required_columns.issubset(df.columns):
        print("Error: The fetched data does not contain required columns 'date' and 'discharge'.")
        return None

    return df

def validate_data(df: pd.DataFrame):
    """
    Validate a pandas DataFrame containing 'date' and 'discharge' columns with extensive error handling.

    The function checks if the DataFrame is None or empty, verifies required columns, and prints an error message if any check fails.
    It checks for missing values in 'date' and 'discharge' columns, prints a summary of missing values, and removes rows with missing values.

    Parameters:
        df (pd.DataFrame): DataFrame to validate.

    Returns:
        pd.DataFrame: Cleaned DataFrame after removing rows with missing values, or None if validation fails.
    """
    if df is None or df.empty:
        print("Error: The DataFrame is either None or empty.")
        return None

    # Check if required columns exist
    required_columns = ['date', 'discharge']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' is missing in the DataFrame.")
            return None

    try:
        missing_date = df['date'].isnull().sum()
        missing_discharge = df['discharge'].isnull().sum()
    except Exception as e:
        print(f"Error checking for missing values: {e}")
        return None

    print(f"Missing values summary - date: {missing_date}, discharge: {missing_discharge}")

    if missing_date > 0 or missing_discharge > 0:
        try:
            df = df.dropna(subset=['date', 'discharge'])
            print("Rows with missing 'date' or 'discharge' have been removed.")
        except Exception as e:
            print(f"Error removing rows with missing values: {e}")
            return None

    return df

def calculate_annual_indicators(df: pd.DataFrame):
    """
    Calculate annual hydrological indicators from daily streamflow data.

    The function performs the following steps:
      1. Ensures the 'date' column is datetime and adds a 'year' column.
      2. For each year, calculates:
         a. Mean annual flow (mean of 'discharge').
         b. Annual 7-day low flow: computed as the minimum of the 7-day rolling average of 'discharge'.
         c. Date of the annual peak flow: identified by the maximum 'discharge' and its corresponding 'date'.
      3. Combines these metrics into a summary DataFrame with columns: 'year', 'mean_flow', '7day_low_flow', and 'peak_flow_date'.

    Parameters:
        df (pd.DataFrame): Validated daily streamflow DataFrame with 'date' and 'discharge' columns.

    Returns:
        pd.DataFrame: Summary DataFrame with annual indicators.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    summary_records = []

    for year, group in df.groupby('year'):
        group = group.sort_values('date')
        mean_flow = group['discharge'].mean()
        # Compute 7-day rolling average and then get the minimum value as 7-day low flow.
        rolling_avg = group['discharge'].rolling(window=7).mean()
        seven_day_low_flow = rolling_avg.min()
        # Identify the date of the annual peak flow.
        idx_max = group['discharge'].idxmax()
        peak_flow_date = group.loc[idx_max, 'date']
        summary_records.append({
            'year': year,
            'mean_flow': mean_flow,
            '7day_low_flow': seven_day_low_flow,
            'peak_flow_date': peak_flow_date
        })

    summary_df = pd.DataFrame(summary_records)
    return summary_df

def plot_daily_hydrograph(daily_df: pd.DataFrame):
    """
    Plot a daily hydrograph from a DataFrame containing 'date' and 'discharge' columns.
    The function validates the DataFrame, plots a line graph, labels axes, adds title and grid,
    applies tight layout, saves the plot as 'daily_hydrograph.png', and closes the plot.
    
    Parameters:
        daily_df (pd.DataFrame): DataFrame with daily streamflow data.
    """
    try:
        if daily_df is None or daily_df.empty:
            raise ValueError("Input daily_df is None or empty.")
        if not {'date', 'discharge'}.issubset(daily_df.columns):
            raise ValueError("DataFrame missing required columns 'date' and 'discharge'.")
        plt.figure(figsize=(10,5))
        plt.plot(daily_df['date'], daily_df['discharge'], label='Discharge', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Discharge (cfs)')
        plt.title('Daily Hydrograph')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('daily_hydrograph.png')
        plt.close()
    except Exception as e:
        print(f"Error in plot_daily_hydrograph: {e}")

def plot_annual_summary(annual_df: pd.DataFrame):
    """
    Create a visualization of annual hydrological indicators.
    The visualization includes a bar chart of mean annual flow with an overlay of the 7-day low flow values.
    Labels axes, adds legends and title, saves the figure as 'annual_summary.png', and closes the plot.
    
    Parameters:
        annual_df (pd.DataFrame): DataFrame with annual summary indicators.
    """
    try:
        if annual_df is None or annual_df.empty:
            raise ValueError("Input annual_df is None or empty.")
        required_columns = {'year', 'mean_flow', '7day_low_flow'}
        if not required_columns.issubset(annual_df.columns):
            raise ValueError("DataFrame missing one or more required columns: 'year', 'mean_flow', '7day_low_flow'.")
        
        plt.figure(figsize=(10,6))
        # Bar chart for mean annual flow
        plt.bar(annual_df['year'].astype(str), annual_df['mean_flow'], color='skyblue', label='Mean Annual Flow')
        # Overlay line plot for 7-day low flow
        plt.plot(annual_df['year'].astype(str), annual_df['7day_low_flow'], marker='o', color='red', label='7-Day Low Flow')
        plt.xlabel('Year')
        plt.ylabel('Flow (cfs)')
        plt.title('Annual Mean Flow and 7-Day Low Flow Summary')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('annual_summary.png')
        plt.close()
    except Exception as e:
        print(f"Error in plot_annual_summary: {e}")

if __name__ == "__main__":
    print("Starting experiment...")
    usgs_site_number = "05454500"
    start_date_str = "2019-01-01"
    end_date_str = "2023-12-31"
    df = fetch_streamflow_data(usgs_site_number, start_date_str, end_date_str)
    if df is not None and not df.empty:
        print(f"Number of records fetched: {len(df)}")
    else:
        print("No data fetched.")
    df = validate_data(df)
    if df is not None:
        annual_indicators = calculate_annual_indicators(df)
        plot_daily_hydrograph(df)
        plot_annual_summary(annual_indicators)
    else:
        print("Data validation failed.")
    print("Experiment completed.")