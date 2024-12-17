
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from typing import Dict, Any

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV (ToDo: update to web scrape)
    Input: filepath as a string
    Output: data frame    
    """
    # Seasons based on dates
    seasons = {
        "2024/25": (pd.to_datetime("2024-09-01"), pd.to_datetime("2025-07-01")),
        "2023/24": (pd.to_datetime("2023-09-01"), pd.to_datetime("2024-07-01")),
        "2022/23": (pd.to_datetime("2022-09-01"), pd.to_datetime("2023-07-01")),
        "2021/22": (pd.to_datetime("2021-09-01"), pd.to_datetime("2022-07-01")),
        "2020/21": (pd.to_datetime("2020-09-01"), pd.to_datetime("2021-07-01")),
        "2019/20": (pd.to_datetime("2019-09-01"), pd.to_datetime("2020-07-01")),
        "2018/19": (pd.to_datetime("2018-09-01"), pd.to_datetime("2019-07-01")),
        "2017/18": (pd.to_datetime("2017-09-01"), pd.to_datetime("2018-07-01")),
        "2016/17": (pd.to_datetime("2016-09-01"), pd.to_datetime("2017-07-01")),
        "2015/16": (pd.to_datetime("2015-09-01"), pd.to_datetime("2016-07-01")),
        "2014/15": (pd.to_datetime("2014-09-01"), pd.to_datetime("2015-07-01")),
        "2013/14": (pd.to_datetime("2013-09-01"), pd.to_datetime("2014-07-01")),
        "2012/13": (pd.to_datetime("2012-09-01"), pd.to_datetime("2013-07-01")),
        "2011/12": (pd.to_datetime("2011-09-01"), pd.to_datetime("2012-07-01")),
        "2010/11": (pd.to_datetime("2010-09-01"), pd.to_datetime("2011-07-01")),
        }
    
    # Load data with efficient data types
    df = pd.read_csv(filepath, parse_dates=['Date'],
                     dtype={
                         'Trigger': 'category',
                         'Region': 'category',
                         'Weak Layer': 'category'
                     })
    
    # Select relevant columns
    columns_to_keep = ['Date', 'Region', 'Place', 'Trigger', 'Trigger: additional info',
       'Weak Layer', 'Depth', 'Width', 'Vertical', 'Aspect', 'Elevation',
       'Coordinates', 'Caught', 'Carried', 'Buried - Partly', 'Buried - Fully',
       'Injured', 'Killed']
    df = df[columns_to_keep]
    
    # Clean / convert numeric columns
    numeric_columns = ['Depth', 'Width', 'Vertical', 'Elevation', 
                       'Caught', 'Carried', 'Buried - Partly', 'Buried - Fully']
    
    for col in numeric_columns:
        if df[col].dtype == 'object':  
            df[col] = df[col].astype(str).str.replace(',', '').str.replace("'", '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Determine season and add as new column
    def determine_season(date):
        for season, (start, end) in seasons.items():
            if start <= date <= end:
                return season
        return 'unknown'
    df['Season'] = df['Date'].apply(determine_season)

    # Collapsing Elevation levels into one column
    def categorize_elevation(elevation):
        if pd.isna(elevation):
            return 'unknown'
        elif elevation > 9000:
            return ">9K"
        elif 8000 <= elevation <= 9000: 
            return ">8K and <9K"
        else:
            return "<8K"
    df['Elevation Category'] = df['Elevation'].apply(categorize_elevation)

    return df

def determine_weak_layer(df: pd.DataFrame, year: str) -> pd.DataFrame:
    """Create dataframe of weak layer for chart"""

    weak_layer = df[['Season', 'Weak Layer']]
    weak_layer = weak_layer[weak_layer['Season'] == year]
    weak_layer_counts = weak_layer['Weak Layer'].value_counts()
    return weak_layer_counts

@st.cache_data
def cached_load_and_preprocess_data(filepath):
    """Leveragin streamlits cached method of loading and preprocessing data using pre-defined function"""
    return load_and_preprocess_data(filepath)

def main():
    """Main function for everything else"""

    # Config Streamlit page
    st.set_page_config(
        layout='wide',
        page_icon='V',
        menu_items={
            'Get help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': 'https://www.extremelycoolapp.com/bug',
            'About': '# This is a header. This is an *extremely* cool app!'
            }
        )    

    # Pull in data; update to non-hardcoded 
    filepath = '/Users/chrishyatt/Library/Mobile Documents/com~apple~CloudDocs/Projects/gh_repos/AvyDash/avalanches.csv'

    # Call function to Load / prepocess data  
    df = cached_load_and_preprocess_data(filepath)

    year = '2012/13'

    weak_layer_counts = determine_weak_layer(df, year)

    plt.figure(figsize=(8,5))
    plt.bar(weak_layer_counts.index, weak_layer_counts.values, color='skyblue', edgecolor='black')

    plt.title("Weak Layer Distribution", fontsize=16)
    plt.xlabel("Weak Layer", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Rotate labels for readability

    # Show plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()