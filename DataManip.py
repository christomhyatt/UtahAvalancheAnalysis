import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

filepath = '/Users/chrishyatt/Library/Mobile Documents/com~apple~CloudDocs/Projects/gh_repos/AvyDash/avalanches.csv'
avy_csv_raw = pd.read_csv(filepath)
df = pd.DataFrame(avy_csv_raw)

# Pulling relevant columns
df = df[['Date', 'Region', 'Place', 'Trigger', 'Trigger: additional info',
       'Weak Layer', 'Depth', 'Width', 'Vertical', 'Aspect', 'Elevation',
       'Coordinates', 'Caught', 'Carried', 'Buried - Partly', 'Buried - Fully',
       'Injured', 'Killed']]

## Converting Date column in a Winter Season column ##
df['Date'] = pd.to_datetime(df['Date'])

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

## Avalanche Sizes ##
avy_size = df.copy()  # Ensure it's a standalone DataFrame
avy_size = df[['Season', 'Depth', 'Width', 'Vertical', 'Weak Layer']]
# Clean and convert Elevation column to numerical
avy_size.loc[:,'Depth'] = avy_size['Depth'].replace(regex=["'"], value='')
avy_size.loc[:, 'Width'] = avy_size['Width'].replace(regex=["'"], value='')
avy_size.loc[:, 'Vertical'] = avy_size['Vertical'].replace(regex=["'"], value='')

avy_size.loc[:, 'Depth'] = pd.to_numeric(avy_size['Depth'], errors="coerce")
avy_size.loc[:, 'Width'] = pd.to_numeric(avy_size['Width'], errors="coerce")
avy_size.loc[:, 'Vertical'] = pd.to_numeric(avy_size['Vertical'], errors="coerce")
avy_size = avy_size.dropna()
print(avy_size[avy_size['Depth'] > 2])
# print(avy_size.dtypes)
