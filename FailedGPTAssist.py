
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

def get_elevation_distribution(df: pd.DataFrame, year: str) -> Dict[str, Dict[str, Any]]:
    """Clean elevation data, determine elevation category and add as new column
    Input: cleaned dataframe and selected year by user
    Output: dictionary of elevation counts into three categories (top, middle, bottom)"""

    elevation_data = (
        df[df['Season'] == year]
        .groupby('Elevation Category')
        .size()
        .reindex(['>9K', '>8K and <9K', '>8K'], fill_value=0))
    
    # Test when finished*********** .size()
    return {
        'top' : {'range': '>9k', 'count': elevation_data.get(">9K",0)},
        'middle' : {'range': '>8K and <9K', 'count': elevation_data.get('>8K and <9K',0)},
        'bottom' : {'range': '>8K', 'count': elevation_data.get('>8K',0)}
    }

def process_avalanche_outcomes(df: pd.DataFrame, year:str) -> pd.DataFrame:
    """Clean avalanche outcomes, count, and add new column
    Input: cleaned dataframe and selected year by user
    Output: dataframe"""

    # Filter for user specified year
    filtered_df = df[df['Season'] == year]

    # Create dictionary for outcomes when != 0
    outcome_metrics = {
        'Caught': filtered_df[filtered_df['Caught'] > 0],
        'Carried': filtered_df[filtered_df['Carried'] > 0],
        'Buried - Partly': filtered_df[filtered_df['Buried - Partly'] > 0],
        'Buried - Fully': filtered_df[filtered_df['Buried - Fully'] > 0 ]
    }

    # Aggregate outcomes
    outcome_summary = []
    for outcome, data in outcome_metrics.items():
        summary = data[['Depth', 'Width', 'Vertical']].mean().round(1)
        summary['Outcome'] = outcome
        outcome_summary.append(summary)

    return pd.DataFrame(outcome_summary).set_index('Outcome')

def create_elevation_svg(data: Dict[str, Dict[str, Any]]) -> str:
    """Generate the image (svg) for the elevation distribution
    Input: elevation dictionary defined above
    Output: SVG triangle mountain"""
    
    return f"""
    <svg width="350" height="200" xmlns="http://www.w3.org/2000/svg">
        <polygon points="100,10 120,70 80,70" style="fill:red;stroke:black;stroke-width:3"/>
        <text x="70" y="50" fill="white" font-size="14" font-family="Arial" text-anchor="middle">9k+</text>
        <text x="175" y="50" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            {data['top']['count']} avalanches
        </text>
        <polygon points="80,70 120,70 140,140 60,140" style="fill:orange;stroke:black;stroke-width:3"/>
        <text x="50" y="110" fill="white" font-size="14" font-family="Arial" text-anchor="middle">8-9k</text>
        <text x="190" y="110" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            {data['middle']['count']} avalanches
        </text>
        <polygon points="60,140 140,140 160,200 40,200" style="fill:green;stroke:black;stroke-width:3"/>
        <text x="30" y="180" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            <8k
        </text>
        <text x="210" y="180" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            {data['bottom']['count']} avalanches
        </text>
    </svg>
    """

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

    # List of all seasons for filter / drop down
    seasons = sorted(df['Season'].unique(), reverse=True)

    # Page layout (header / filter)
    col_header_1, col_header_2 = st.columns([.9,.2], gap="small", vertical_alignment="center")
    
    with col_header_2:
        year = st.selectbox('', seasons, placeholder="Select a season", label_visibility="collapsed", index=0)
    
    with col_header_1:
        st.title(f"_{year}_ Avalanches at a Glance")

    # Dashboard layout (charts)
    col1, col2 = st.columns(2, gap="small", vertical_alignment="center")

    with col1:
        # Elevation Distribution
        elevation_data = get_elevation_distribution(df, year)
        st.markdown(f"""
        <div style="text-align: center;">
            {create_elevation_svg(elevation_data)}
        """, unsafe_allow_html=True)

    with col2: 
        # Avalanche
        outcomes_data = process_avalanche_outcomes(df, year)

        # Grouped bar chart for avalanch sizes
        fig, ax = plt.subplots(figsize=(6,5))
        outcomes = outcomes_data.index.tolist()
        metrics = ['Depth', 'Width', 'Vertical']

        width = 0.2
        x = range(len(outcomes))

        for i, metric in enumerate(metrics):
            ax.bar(
                [pos + i * width for pos in x],
                outcomes_data[metric],
                width=width,
                label=metric,
            )
        
        ax.set_title(f"Avalanche Outcomes by Sizes")
        ax.set_xlabel("Outcomes")
        ax.set_ylabel("Inches")
        ax.set_xticks([pos + width for pos in x])
        ax.set_xticklabels(outcomes)
        ax.legend(title="Metrics")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        st.pyplot(fig)

if __name__ == "__main__":
    main()