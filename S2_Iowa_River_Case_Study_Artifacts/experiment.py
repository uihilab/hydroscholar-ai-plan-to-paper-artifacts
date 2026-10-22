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
from scipy import stats # Commonly used, include by default
from io import StringIO # Useful for parsing text data
import datetime # Useful for date handling

# --- Optional Imports (Uncomment as needed) ---
# import pymannkendall as mk
# from scipy.optimize import curve_fit
# from scipy.stats import pearson3, skew, norm

# --- Environment Setup (Example) ---
# Load API keys or configurations if necessary
# noaa_token = os.environ.get('NOAA_TOKEN')
# openai_api_key = os.environ.get('OPENAI_API_KEY')

# --- Function Definitions Will Be Added Here By AI --- 
def fetch_streamflow_data(usgs_site_number: str, start_date_str: str, end_date_str: str):
    """
    Fetch streamflow data from the USGS NWIS DV service.
    Returns a pandas DataFrame with columns 'date' and 'streamflow' if successful, or None if an error occurs.
    """
    base_url = 'https://waterservices.usgs.gov/nwis/dv/'
    params = {
        'format': 'json',
        'sites': usgs_site_number,
        'startDT': start_date_str,
        'endDT': end_date_str,
        'parameterCd': '00060',
        'siteStatus': 'all'
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        json_data = response.json()
        # Navigate the nested JSON structure
        time_series = json_data['value']['timeSeries'][0]
        records = time_series['values'][0]['value']
        dates = [record['dateTime'] for record in records]
        values = []
        for record in records:
            try:
                value = float(record['value'])
            except (ValueError, TypeError):
                value = None
            values.append(value)
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'streamflow': values
        })
        return df
    except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError):
        return None

def validate_streamflow_data(df: pd.DataFrame):
    """
    Validates the streamflow data contained in the pandas DataFrame.
    1. Checks if the DataFrame is None or empty; if so, prints an error message and returns None.
    2. Verifies the presence of the 'date' and 'streamflow' columns.
    3. Checks for missing (NaN) values in 'streamflow'. If found, prints a warning with the count and percentage of missing records.
    4. Returns the DataFrame unchanged for further processing.
    """
    if df is None or df.empty:
        print("Error: The provided DataFrame is None or empty.")
        return None
    
    required_columns = ['date', 'streamflow']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}'.")
            return None
    
    total_records = len(df)
    missing_count = df['streamflow'].isna().sum()
    if missing_count > 0:
        missing_percent = (missing_count / total_records) * 100
        print(f"Warning: Found {missing_count} missing 'streamflow' records ({missing_percent:.2f}% of total).")
    
    return df

def calculate_annual_indicators(df: pd.DataFrame):
    """
    Calculates annual hydrological indicators from daily streamflow data.
    
    Steps:
    1. Ensures the 'date' column is in datetime format and extracts a new 'year' column.
    2. For each year, calculates:
       (a) Mean annual flow from the average of 'streamflow'.
       (b) Annual 7-day low flow by computing a rolling mean (window=7, min_periods=7) over 'streamflow'
           and selecting the minimum value.
       (c) Date of annual peak flow by identifying the date corresponding to the maximum 'streamflow'.
    3. Assembles these indicators into a new DataFrame with columns:
       'year', 'mean_annual_flow', 'annual_7day_low_flow', and 'peak_flow_date'.
    4. Returns the new DataFrame.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['rolling_mean'] = df['streamflow'].rolling(window=7, min_periods=7).mean()
    
    records = []
    for year, group in df.groupby('year'):
        mean_flow = group['streamflow'].mean()
        annual_7day_low_flow = group['rolling_mean'].min()
        idx_peak = group['streamflow'].idxmax()
        peak_date = group.loc[idx_peak, 'date'] if pd.notnull(idx_peak) else None
        records.append({
            'year': year,
            'mean_annual_flow': mean_flow,
            'annual_7day_low_flow': annual_7day_low_flow,
            'peak_flow_date': peak_date
        })
    
    result_df = pd.DataFrame(records)
    return result_df

def plot_daily_hydrograph(df: pd.DataFrame):
    """
    Plots a daily streamflow hydrograph from the provided DataFrame.
    
    Steps:
    1. Checks if the DataFrame is valid (not None or empty).
    2. Plots a time series with 'date' on the x-axis and 'streamflow' on the y-axis.
    3. Sets axis labels ('Date', 'Streamflow (cfs)'), title, and grid.
    4. Optimizes layout with plt.tight_layout(), saves the figure as 'daily_hydrograph.png', and calls plt.close().
    5. Prints a confirmation message upon successful saving.
    """
    if df is None or df.empty:
        print("Error: Cannot plot hydrograph because the DataFrame is None or empty.")
        return
    
    plt.figure()
    plt.plot(df['date'], df['streamflow'], marker='o', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Streamflow (cfs)')
    plt.title('Daily Streamflow Hydrograph: Iowa River (2019-2023)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('daily_hydrograph.png')
    plt.close()
    print("Hydrograph saved as 'daily_hydrograph.png'.")

def plot_annual_indicators(indicators_df: pd.DataFrame):
    """
    Plots annual hydrological indicators from the provided DataFrame.
    
    Steps:
    1. Verifies that the provided DataFrame is valid and non-empty.
    2. Creates a plot displaying annual mean flow and annual 7-day low flow as lines.
    3. Overlays text annotations with the date of the annual peak flow for each year.
    4. Adds axis labels, legend, title, and grid.
    5. Optimizes the layout, saves the plot as 'annual_indicators_summary.png', and closes the plot.
    6. Prints a confirmation message post-saving.
    """
    if indicators_df is None or indicators_df.empty:
        print("Error: Provided annual indicators DataFrame is None or empty.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    years = indicators_df['year']
    mean_flow = indicators_df['mean_annual_flow']
    low_flow = indicators_df['annual_7day_low_flow']
    
    ax.plot(years, mean_flow, marker='o', linestyle='-', label='Mean Annual Flow')
    ax.plot(years, low_flow, marker='s', linestyle='-', label='Annual 7-day Low Flow')
    
    for _, row in indicators_df.iterrows():
        peak_date = row['peak_flow_date']
        if pd.notnull(peak_date):
            annotation = peak_date.strftime('%Y-%m-%d')
        else:
            annotation = 'N/A'
        y_position = max(row['mean_annual_flow'], row['annual_7day_low_flow'])
        ax.text(row['year'], y_position, annotation, ha='center', va='bottom', fontsize=8, color='black')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Flow (cfs)')
    ax.set_title('Annual Hydrologic Indicators for Iowa River (2019-2023)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('annual_indicators_summary.png')
    plt.close()
    print("Annual indicators summary saved as 'annual_indicators_summary.png'.")

if __name__ == "__main__":
    print("Starting experiment...")
    usgs_site_number = '05454500'
    start_date_str = '2019-01-01'
    end_date_str = '2023-12-31'
    
    print("Fetching streamflow data...")
    df = fetch_streamflow_data(usgs_site_number, start_date_str, end_date_str)
    if df is not None:
        print(f"Fetched {len(df)} records.")
    else:
        print("Failed to fetch streamflow data.")
    
    print("Validating streamflow data...")
    df = validate_streamflow_data(df)
    if df is None:
        print("Data validation failed.")
        sys.exit(1)
    
    print("Calculating annual indicators...")
    annual_indicators = calculate_annual_indicators(df)
    
    print("Generating daily hydrograph plot...")
    plot_daily_hydrograph(df)
    
    print("Generating annual indicators plot...")
    plot_annual_indicators(annual_indicators)
    
    print("Experiment finished.")