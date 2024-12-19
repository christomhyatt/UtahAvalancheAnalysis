import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import altair as alt

## Config Streamlit page
st.set_page_config(
    layout="wide",
    page_icon="🏔️",
    menu_items={
        'Get help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
        })

## Pull in data
filepath = '/Users/chrishyatt/Library/Mobile Documents/com~apple~CloudDocs/Projects/gh_repos/AvyDash/avalanches.csv'
avy_csv_raw = pd.read_csv(filepath)
df = pd.DataFrame(avy_csv_raw) 

df = df[['Date', 'Region', 'Place', 'Trigger', 'Trigger: additional info',
       'Weak Layer', 'Depth', 'Width', 'Vertical', 'Aspect', 'Elevation',
       'Coordinates', 'Caught', 'Carried', 'Buried - Partly', 'Buried - Fully',
       'Injured', 'Killed']]

## Convert Date column in a Winter Seasin column
df['Date'] = pd.to_datetime(df['Date'])

## Creating the season column based on the conditions
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

# Creating the season column based on the conditions
for season, (start_date, end_date) in seasons.items():
    df[season] = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df[season] = df[season].apply(lambda x: season if x else "Unknown")

df['Season'] = df[seasons.keys()].apply(lambda row: next((season for season in row if season != 'Unknown'), 'Unknown'), axis=1)
df = df.drop(columns=seasons.keys())

# title and filter page formatting
col_header_1, col_header_2 = st.columns([.9,.2], gap="small", vertical_alignment="center",)
with col_header_2:
    year = st.selectbox('',(seasons.keys()), placeholder="Select a season", label_visibility="collapsed", index=0)
with col_header_1:
    st.title(f"_{year}_ Avalanches at a Glance")

# dashboards
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")

with col1:
    ## Elevation Data Only
    avy_elevation = df.copy()
    avy_elevation = df[['Elevation', 'Season']]

    # Clean and convert Elevation column to numerical
    avy_elevation['Elevation'] = avy_elevation['Elevation'].replace(regex=[',', "'"], value='')
    avy_elevation['Elevation Detail'] = pd.to_numeric(avy_elevation['Elevation'], errors="coerce")

    # Defining Elevation levels
    avy_elevation['Top'] = avy_elevation['Elevation Detail'] > 9000
    avy_elevation['Middle'] = (avy_elevation['Elevation Detail'] < 9000) & (avy_elevation['Elevation Detail'] > 8000)
    avy_elevation['Bottom'] = avy_elevation['Elevation Detail'] < 8000

    # Collapsing Elevation levels into one column
    def determine_elevation(row):
        if row['Top']:
            return ">9K"
        elif row['Middle']:
            return ">8K and <9K"
        elif row['Bottom']:
            return "<8K"
        else:
            return "Unknown"

    avy_elevation['Elevation'] = avy_elevation.apply(determine_elevation, axis=1)
    avy_elevation = avy_elevation.drop(columns=['Top', 'Middle', 'Bottom', 'Elevation Detail'])

    avy_elevation = avy_elevation.groupby(['Elevation', 'Season']).size().reset_index(name='Count')
    
    # Function to safely extract data or return a default
    def safe_extract_data(df, row_index, col_range, col_count):
        if not df.empty:
            try:
                return {"range": df.iloc[row_index, col_range], "count": df.iloc[row_index, col_count]}
            except IndexError:
                return {"range": "N/A", "count": 0}  # Default values if index is out of range
        else:
            return {"range": "N/A", "count": 0}  # Default values if DataFrame is empty

    top_data = avy_elevation[(avy_elevation['Elevation'] == '>9K') & (avy_elevation['Season'] == year)]
    middle_data = avy_elevation[(avy_elevation['Elevation'] == '>8K and <9K') & (avy_elevation['Season'] == year)]
    bottom_data = avy_elevation[(avy_elevation['Elevation'] == '<8K') & (avy_elevation['Season'] == year)]

    svg_data = {
        "top": safe_extract_data(top_data, 0, 0, 2),
        "middle": safe_extract_data(middle_data, 0, 0, 2),
        "bottom": safe_extract_data(bottom_data, 0, 0, 2)
    }

    # SVG with embedded text
    svg_code = f"""
    <svg width="350" height="200" xmlns="http://www.w3.org/2000/svg">
        <!-- Top section -->
        <polygon id="top" points="100,10 120,70 80,70" style="fill:red;stroke:black;stroke-width:3"/>
        <text x="70" y="50" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            9k+ 
        </text>
        <text x="175" y="50" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            {svg_data['top']['count']} avalanches
        </text>
        <!-- Middle section -->
        <polygon id="middle" points="80,70 120,70 140,140 60,140" style="fill:orange;stroke:black;stroke-width:3"/>
        <text x="50" y="110" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            8-9k
        </text>
        <text x="190" y="110" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            {svg_data['middle']['count']} avalanches
        </text>
        <!-- Bottom section -->
        <polygon id="bottom" points="60,140 140,140 160,200 40,200" style="fill:green;stroke:black;stroke-width:3"/>
        <text x="30" y="180" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            <8k
        </text>
        <text x="210" y="180" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            {svg_data['bottom']['count']} avalanches
        </text>
    </svg>
    """

    # Render SVG in Streamlit
    st.subheader('Avalanches by Elevation (ft)')

    st.markdown(f"""
    <div style="text-align: center;">
        {svg_code}
    </div>
    """, unsafe_allow_html=True)

    ### Second Chart in Column 1
    st.markdown('####')
    st.subheader('Weak Layer Distribution')
    weak_layer = df.copy()
    weak_layer = df[['Season', 'Weak Layer']]
    weak_layer = weak_layer[weak_layer['Season'] == year]
    weak_layer_counts = weak_layer['Weak Layer'].value_counts()

    fig, ax = plt.subplots(figsize=(8,5))
    wl_bars = ax.bar(weak_layer_counts.index, weak_layer_counts.values, color='skyblue', edgecolor='black')

    for bar in wl_bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom')

    ax.set_ylabel("Avalanches", fontsize=12)
    ax.set_xticklabels(weak_layer_counts.index, rotation=45, ha='right', fontsize=8)  # Rotate labels for readability

    st.pyplot(fig)

with col2:
    st.subheader('Human Triggered Avalanche Outcomes by Size')
    ## Avalanche Sizes ## % of people caught in avalanches by size (Caught, Carried, Burried, Killed groups)
    avy_size = df.copy()  # Ensure it's a standalone DataFrame to avoid warnings in console
    avy_size = df[['Season', 'Depth', 'Width', 'Vertical', 'Caught', 'Carried', 'Buried - Partly', 'Buried - Fully']]

    # Clean and convert Depth, Width, Vertical columns to numerical
    avy_size.loc[:,'Depth'] = avy_size['Depth'].replace(regex=["'"], value='')
    avy_size.loc[:, 'Depth'] = round(pd.to_numeric(avy_size['Depth'], errors="coerce"),1)

    avy_size.loc[:, 'Width'] = avy_size['Width'].replace(regex=["'"], value='')
    avy_size.loc[:, 'Width'] = round(pd.to_numeric(avy_size['Width'], errors="coerce"),1)

    avy_size.loc[:, 'Vertical'] = avy_size['Vertical'].replace(regex=["'"], value='')
    avy_size.loc[:, 'Vertical'] = round(pd.to_numeric(avy_size['Vertical'], errors="coerce"),1)

    # Clean and convert Caught, Carried, Buried columns to numerical
    avy_size.loc[:, 'Caught'] = round(pd.to_numeric(avy_size['Caught'], errors="coerce"),1)
    avy_size.loc[:, 'Carried'] = round(pd.to_numeric(avy_size['Carried'], errors="coerce"),1)
    avy_size.loc[:, 'Buried - Partly'] = round(pd.to_numeric(avy_size['Buried - Partly'], errors="coerce"),1)
    avy_size.loc[:, 'Buried - Fully'] = round(pd.to_numeric(avy_size['Buried - Fully'], errors="coerce"),1)

    avy_size = avy_size.dropna(subset=['Depth', 'Width', 'Vertical'])

    ## Filter the year
    filtered_avy_size = avy_size.copy()
    filtered_avy_size = avy_size[avy_size['Season'] == year]
    # print(filtered_avy_size.head())

    # caught = season, AVG(depth), AVG(width), AVG(vertical)
    caught_data = filtered_avy_size[['Season', 'Depth', 'Width', 'Vertical', 'Caught']]
    caught_data = caught_data[caught_data['Caught'] > 0]
    caught_data = caught_data.groupby('Season').mean().round(1)
    caught_data['Outcome'] = 'Caught'
    # print(caught_data.head())

    # carried = season, AVG(depth), AVG(width), AVG(vertical)
    carried_data = filtered_avy_size[['Season', 'Depth', 'Width', 'Vertical', 'Carried']]
    carried_data = carried_data[carried_data['Carried'] > 0]
    carried_data = round(carried_data.groupby('Season').mean(),1)
    carried_data['Outcome'] = 'Carried'
    # print(carried_data.head())

    # Buried - Partly = season, AVG(depth), AVG(width), AVG(vertical)
    partly_burried_data = filtered_avy_size[['Season', 'Depth', 'Width', 'Vertical', 'Buried - Partly']]
    partly_burried_data = partly_burried_data[partly_burried_data['Buried - Partly'] > 0]
    partly_burried_data = round(partly_burried_data.groupby('Season').mean(),1)
    partly_burried_data['Outcome'] = 'Buried - Partly'
    # print(partly_burried_data.head())

    # carried = season, AVG(depth), AVG(width), AVG(vertical)
    fully_burried_data = filtered_avy_size[['Season', 'Depth', 'Width', 'Vertical', 'Buried - Fully']]
    fully_burried_data = fully_burried_data[fully_burried_data['Buried - Fully'] > 0]
    fully_burried_data = round(fully_burried_data.groupby('Season').mean(),1)
    fully_burried_data['Outcome'] = 'Buried - Fully'
    # print(fully_burried_data.head())

    # Append DFs into one 
    full_dataset = pd.concat([caught_data, carried_data, partly_burried_data, fully_burried_data])
    full_dataset = full_dataset.drop(columns = [ 'Caught', 'Carried', 'Buried - Partly', 'Buried - Fully'])
    # full_dataset = full_dataset.set_index('Season')

    ## Chart data
    outcomes = full_dataset['Outcome'].values
    metrics = ['Depth', 'Width', 'Vertical']
    values = full_dataset[metrics].T

    fig, ax = plt.subplots(figsize=(6,5))
    x = range(len(outcomes))
    width = .2

    for i, metric in enumerate(metrics):
        ax.bar(
            [pos + i * width for pos in x],  # Offset each group by bar width
            values.iloc[i],  # Values for this outcome
            width=width,
            label=metric,
        )
    ax.set_title(f"Avalanche Outcomes by Sizes ")
    ax.set_xlabel("Outcomes")
    ax.set_ylabel("Inches")
    ax.set_xticks([pos + width for pos in x])
    ax.set_xticklabels(outcomes)
    ax.legend(title="Metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    st.pyplot(fig)

    ### Second Chart in Column 2
    st.markdown('####')
    st.subheader('Avalanches by Aspect & Elevation Rose')