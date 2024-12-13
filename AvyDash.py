import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

## Config Streamlit page
st.set_page_config(
    layout="wide",
    page_icon="V",
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

for season, (start_date, end_date) in seasons.items():
    df[season] = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df[season] = df[season].apply(lambda x: season if x else "Unknown")

df['Season'] = df[seasons.keys()].apply(lambda row: next((season for season in row if season != 'Unknown'), 'Unknown'), axis=1)
df = df.drop(columns=seasons.keys())

## Elevation Data Only
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

summary = avy_elevation.groupby(['Elevation', 'Season']).size().reset_index(name='Count')

# title and filter page formatting
col_header_1, col_header_2 = st.columns([.9,.2], gap="small", vertical_alignment="center",)
with col_header_2:
    year = st.selectbox('',(seasons.keys()), placeholder="Select a season", label_visibility="collapsed", index=0)
with col_header_1:
    st.title(f"_{year}_ Avalanches at a Glance")

# dashboards
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")

with col1:
    # Function to safely extract data or return a default
    def safe_extract_data(df, row_index, col_range, col_count):
        if not df.empty:
            try:
                return {"range": df.iloc[row_index, col_range], "count": df.iloc[row_index, col_count]}
            except IndexError:
                return {"range": "N/A", "count": 0}  # Default values if index is out of range
        else:
            return {"range": "N/A", "count": 0}  # Default values if DataFrame is empty

    top_data = summary[(summary['Elevation'] == '>9K') & (summary['Season'] == year)]
    middle_data = summary[(summary['Elevation'] == '>8K and <9K') & (summary['Season'] == year)]
    bottom_data = summary[(summary['Elevation'] == '<8K') & (summary['Season'] == year)]

    data = {
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
            {data['top']['count']} avalanches
        </text>
        <!-- Middle section -->
        <polygon id="middle" points="80,70 120,70 140,140 60,140" style="fill:orange;stroke:black;stroke-width:3"/>
        <text x="50" y="110" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            8-9k
        </text>
        <text x="190" y="110" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            {data['middle']['count']} avalanches
        </text>
        <!-- Bottom section -->
        <polygon id="bottom" points="60,140 140,140 160,200 40,200" style="fill:green;stroke:black;stroke-width:3"/>
        <text x="30" y="180" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            <8k
        </text>
        <text x="210" y="180" fill="white" font-size="14" font-family="Arial" text-anchor="middle">
            {data['bottom']['count']} avalanches
        </text>
    </svg>
    """

    # Render SVG in Streamlit
    st.markdown(f"""
    <div style="text-align: center;">
        {svg_code}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.write("column 2 test")