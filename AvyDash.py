import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import altair as alt

# Config Streamlit page
st.set_page_config(
    layout="wide",
    page_icon="🏔️",
    menu_items={
        'About': "All data is gathered from the Utah Avalanch Center (UAC).\
             The UAC tracks all backcountry observations related to Utah's\
             snowpack and avalaches on their website (https://utahavalanchecenter.org/observations).\
             \n\n***This dashboard was created strictly for educational purposes to better understand Utah's snowpack.***"
        })

# Pull in data
filepath = '/Users/chrishyatt/Library/Mobile Documents/com~apple~CloudDocs/Projects/gh_repos/AvyDash/avalanches.csv'
avy_csv_raw = pd.read_csv(filepath)
df = pd.DataFrame(avy_csv_raw) 

df = df[['Date', 'Region', 'Place', 'Trigger', 'Trigger: additional info',
       'Weak Layer', 'Depth', 'Width', 'Vertical', 'Aspect', 'Elevation',
       'Coordinates', 'Caught', 'Carried', 'Buried - Partly', 'Buried - Fully',
       'Injured', 'Killed']]

# Convert Date column in a Winter Seasin column
df['Date'] = pd.to_datetime(df['Date'])

# Creating the season column based on the conditions
seasons = {
    "2024/25": (pd.to_datetime("2024-08-01"), pd.to_datetime("2025-06-01")),
    "2023/24": (pd.to_datetime("2023-08-01"), pd.to_datetime("2024-06-01")),
    "2022/23": (pd.to_datetime("2022-08-01"), pd.to_datetime("2023-06-01")),
    "2021/22": (pd.to_datetime("2021-08-01"), pd.to_datetime("2022-06-01")),
    "2020/21": (pd.to_datetime("2020-08-01"), pd.to_datetime("2021-06-01")),
    "2019/20": (pd.to_datetime("2019-08-01"), pd.to_datetime("2020-06-01")),
    "2018/19": (pd.to_datetime("2018-08-01"), pd.to_datetime("2019-06-01")),
    "2017/18": (pd.to_datetime("2017-08-01"), pd.to_datetime("2018-06-01")),
    "2016/17": (pd.to_datetime("2016-08-01"), pd.to_datetime("2017-06-01")),
    "2015/16": (pd.to_datetime("2015-08-01"), pd.to_datetime("2016-06-01")),
    "2014/15": (pd.to_datetime("2014-08-01"), pd.to_datetime("2015-06-01")),
    "2013/14": (pd.to_datetime("2013-08-01"), pd.to_datetime("2014-06-01")),
    "2012/13": (pd.to_datetime("2012-08-01"), pd.to_datetime("2013-06-01")),
    "2011/12": (pd.to_datetime("2011-08-01"), pd.to_datetime("2012-06-01")),
    "2010/11": (pd.to_datetime("2010-08-01"), pd.to_datetime("2011-06-01")),
    "2009/10": (pd.to_datetime("2009-08-01"), pd.to_datetime("2010-06-01")),
    "2008/09": (pd.to_datetime("2008-08-01"), pd.to_datetime("2009-06-01")),
    "2007/08": (pd.to_datetime("2007-08-01"), pd.to_datetime("2008-06-01")),
    "2006/07": (pd.to_datetime("2006-08-01"), pd.to_datetime("2007-06-01")),
    "2005/06": (pd.to_datetime("2005-08-01"), pd.to_datetime("2006-06-01")),
    "2004/05": (pd.to_datetime("2004-08-01"), pd.to_datetime("2005-06-01")),
    "2003/04": (pd.to_datetime("2003-08-01"), pd.to_datetime("2004-06-01")),
    "2002/03": (pd.to_datetime("2002-08-01"), pd.to_datetime("2003-06-01")),
    "2001/02": (pd.to_datetime("2001-08-01"), pd.to_datetime("2002-06-01")),
    "2000/01": (pd.to_datetime("2000-08-01"), pd.to_datetime("2001-06-01"))
    }

# Creating the season column based on the conditions
for season, (start_date, end_date) in seasons.items():
    df[season] = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df[season] = df[season].apply(lambda x: season if x else "Unknown")

df['Season'] = df[seasons.keys()].apply(lambda row: next((season for season in row if season != 'Unknown'), 'Unknown'), axis=1)
df = df.drop(columns=seasons.keys())

# Title and filter page formatting
col_header_1, col_header_2 = st.columns([.9,.2], gap='small', vertical_alignment="center")
with col_header_2:
    year = st.selectbox('',(seasons.keys()), placeholder='Select a season', label_visibility='collapsed', index=0)
with col_header_1:
    st.title(f'_{year}_ Utah Avalanch & Snowpack')

# High level metrics 
col_metric_1, col_metric_2, col_metric_3, col_metric_4 = st.columns(4, gap='small', vertical_alignment='center')
with col_metric_1:
    num_avalanches = df.copy()
    num_avalanches = num_avalanches[['Season', 'Date']]
    num_avalanches = num_avalanches[num_avalanches['Season'] == year]
    num_avalanches = num_avalanches.count()
    st.metric(label="Avalanches", value=num_avalanches.loc['Season'])
with col_metric_2:
    top_location = df.copy()
    top_location = top_location[['Season', 'Place']]
    top_location = top_location[top_location['Season'] == year]
    top_location = top_location['Place'].value_counts().sort_values(ascending=False)
    top_location_count = top_location.max()
    top_location = top_location.idxmax()
    st.metric(label='Top Location', value=f'{top_location} | {top_location_count}') # Delta from year before!!
with col_metric_3:
    num_fatalities = df.copy()
    num_fatalities = num_fatalities[['Season', 'Killed']]
    num_fatalities = num_fatalities[(num_fatalities['Season'] == year) & (num_fatalities['Killed'] >= 1)]
    num_fatalities = num_fatalities.sum()
    st.metric(label='Fatalities', value=int(num_fatalities.loc['Killed']))
with col_metric_4:
    top_trigger = df.copy()
    top_trigger = top_trigger[['Season', 'Trigger']]
    top_trigger = top_trigger[top_trigger['Season'] == year]
    top_trigger = top_trigger['Trigger'].value_counts().sort_values(ascending=False)
    top_trigger_count = top_trigger.max()
    top_trigger = top_trigger.idxmax()
    st.metric(label='Top Trigger', value=f'{top_trigger} | {top_trigger_count}')

# Defining dashboards columns for charts
col_graph_1, col_graph_2, col_graph_3 = st.columns(3, gap='small', vertical_alignment='center')

with col_graph_1:
    # 1st chart
    st.markdown('##### Avalanches by Aspect & Elevation Rose')

    avy_rose_map = df.copy()
    avy_rose_map = avy_rose_map[avy_rose_map['Season'] == year]
    avy_rose_map = avy_rose_map[['Aspect','Elevation']]

    # Convert elevation from string to numerical feet
    avy_rose_map['Elevation'] = avy_rose_map['Elevation'].replace(regex=[',', "'"], value='')
    avy_rose_map['Elevation'] = pd.to_numeric(avy_rose_map['Elevation'], errors="coerce")

    # Convert aspects to angles (e.g., 70 degrees) for mapping
    aspect_to_angle = {
        'North': 0,
        'Northeast': 45,
        'East': 90,
        'Southeast': 135,
        'South': 180,
        'Southwest': 225,
        'West': 270,
        'Northwest': 315 
    }
    avy_rose_map['Aspect Angle'] = avy_rose_map['Aspect'].map(aspect_to_angle)

    # Convert aspect names to abbreviations
    aspect_to_abbr = {
        'North': 'N',
        'Northeast': 'NE',
        'East': 'E',
        'Southeast': 'SE',
        'South': 'S',
        'Southwest': 'SW',
        'West': 'W',
        'Northwest': 'NW' 
    }

    # Normalize elvation; Convert Elevations so highest elevation = 1 and the lowest elevation = 0
        # Formaula to normalize: (X - Xmin) / (Xmax - Xmin)
    elevation_min, elevation_max = avy_rose_map['Elevation'].min(), avy_rose_map['Elevation'].max()
    avy_rose_map['Normalized Elevation'] = (avy_rose_map['Elevation'] - elevation_min) / (elevation_max - elevation_min)
    avy_rose_map = avy_rose_map.drop(columns=['Elevation', 'Aspect'])

    counts = avy_rose_map.groupby(['Normalized Elevation', 'Aspect Angle']).size().reset_index(name='Count')
    avy_rose_map = (avy_rose_map.merge(counts, on=['Normalized Elevation', 'Aspect Angle'], how='left')).drop_duplicates()

    # Add jitter to avoid overlapping data
    jitter = np.random.normal(0, 3, len(avy_rose_map['Aspect Angle']))
    avy_rose_map['Aspect Angle'] = avy_rose_map['Aspect Angle'] + jitter

    # Set up Polar Coordinate System
    plt.style.use('dark_background')
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,5))

    # Scatter plot with color-mapped Avalanche Count
    cmap = plt.cm.YlOrRd_r # Defining what color scale for the color map (cm)
    norm = Normalize(vmin=avy_rose_map['Count'].min(), vmax=avy_rose_map['Count'].max()) # Normalization for the color scale counts
    colors = cmap(norm(avy_rose_map['Count']))

    scatters = ax.scatter(
        np.radians(avy_rose_map['Aspect Angle']), # Convert data type integer to radian (degrees)
        avy_rose_map['Normalized Elevation'], # Radial distance
        c=colors, # Color of each point
        s=100, # Marker Size
        cmap=cmap,
        alpha=0.7 # Transparency of markers
    ) 

    # Colorbar (legend)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("Avalanches")

    # Customize the plot
    ax.set_theta_zero_location('N') # Sets 0 direction to North
    ax.set_theta_direction(-1) # Sets direction in which the angles increase e.g., Clockwise
    ax.set_xticks(np.radians(list(aspect_to_angle.values())))
    ax.set_xticklabels(list(aspect_to_abbr.values()))

    # Get number of gridlines
    num_gridlines = len(ax.yaxis.get_gridlines())
    
    # Generate elevation labels for each gridline
    elevation_ticks = np.linspace(0, 1, num_gridlines)
    elevation_labels = [f"{int(elevation_min + (e * (elevation_max - elevation_min)))} ft" for e in elevation_ticks]
    ax.set_yticklabels([])  # Clear existing labels
    
    # Custom label positioning
    for idx, label in enumerate(elevation_labels):
        # Skip the last elevation plot; unnecessary
        if idx == len(elevation_labels) - 1:
            continue
        # Position of labels degrees
        angle = -9.8 
        radius = ax.get_yticks()[idx]
        # if radius > 0:  # Don't place label at origin
        ax.text(angle, 
                radius, 
                label,
                ha='center', 
                va='top',
                rotation=0,
                transform=ax.transData)

    # Customize grid lines
    for line in ax.xaxis.get_gridlines():  
        line.set_alpha(0.2)  # Make the angular grid lines more transparent
    for line in ax.yaxis.get_gridlines():  
        line.set_alpha(0.2)  # Make the radial grid lines more transparent

    # Integrate with Streamlit
    st.pyplot(fig)

with col_graph_2:
    # 2nd Chart
    st.markdown('##### Weak Layer Distribution')
    weak_layer = df.copy()
    weak_layer = df[['Season', 'Weak Layer', 'Region']]
    weak_layer = weak_layer[weak_layer['Season'] == year]
    weak_layer = weak_layer[(weak_layer['Weak Layer'].notnull()) & (weak_layer['Region'].notnull())]

    # Creating count column
    counts = weak_layer.groupby(['Weak Layer', 'Region']).size().reset_index(name='Count')
    weak_layer = (weak_layer.merge(counts, on=['Weak Layer', 'Region'], how='left')).drop_duplicates()

    weak_layer_chart = alt.Chart(weak_layer).mark_bar().encode(
        x=alt.X('Weak Layer:N', title=''), # N = Nominal (categorical) data
        y=alt.Y('Count:Q', stack='zero', title=''), # Q = Quantitative data; stacked starting at 0
        color=alt.Color('Region:N', legend=None), # N = Nominal (categorical) data
        tooltip=['Weak Layer', 'Region', 'Count'] # Shows values over hover
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(weak_layer_chart, use_container_width=True,)

with col_graph_3:
    st.markdown('##### Average Size of Human Triggered Avalanches')

    # % of people caught in avalanches by size (Caught, Carried, Burried, Killed groups)
    avy_size = df.copy() 
    avy_size = df[['Season', 'Depth', 'Width', 'Vertical', 'Caught', 'Carried', 'Buried - Partly', 'Buried - Fully']]

    # Create a copy of Depth for ft only; Convert inches to feet; Clean up feet values and remove characters
    def convert_sizes_to_inches(sizes_only):
        for column in sizes_only.columns:
            sizes_only[f'{column} (ft)'] = sizes_only[f'{column}'].copy(deep=True)
            mask = sizes_only[f'{column} (ft)'].str.contains('"', na=False)
            sizes_only.loc[mask, f'{column} (ft)'] = round(sizes_only.loc[mask, f'{column}'].str.replace('"', '').astype(float) / 12, 1)
            mask = sizes_only[f'{column} (ft)'].str.contains("'",na=False)
            sizes_only.loc[mask, f'{column} (ft)'] = sizes_only.loc[mask, f'{column}'].str.replace("'",'').str.replace(',','').astype(float)
        sizes_only.drop(columns=['Depth', 'Width', 'Vertical'], axis=1, inplace=True)
        return sizes_only

    sizes_only = convert_sizes_to_inches(avy_size[['Depth', 'Width', 'Vertical']])
    avy_size = pd.concat([avy_size, sizes_only], axis=1)
    avy_size.drop(columns=['Depth', 'Width', 'Vertical'], axis=1, inplace=True)
    avy_size = avy_size.dropna(subset=['Depth (ft)', 'Width (ft)', 'Vertical (ft)'])

    # Filter the year
    filtered_avy_size = avy_size.copy()
    filtered_avy_size = avy_size[avy_size['Season'] == year]

    caught_data = filtered_avy_size[['Season', 'Depth (ft)', 'Width (ft)', 'Vertical (ft)', 'Caught']]
    caught_data = caught_data[caught_data['Caught'] > 0]
    caught_data = caught_data.groupby('Season').mean().round(1)
    caught_data['Outcome'] = 'Caught'

    carried_data = filtered_avy_size[['Season', 'Depth (ft)', 'Width (ft)', 'Vertical (ft)', 'Carried']]
    carried_data = carried_data[carried_data['Carried'] > 0]
    carried_data = round(carried_data.groupby('Season').mean(),1)
    carried_data['Outcome'] = 'Carried'

    partly_burried_data = filtered_avy_size[['Season', 'Depth (ft)', 'Width (ft)', 'Vertical (ft)', 'Buried - Partly']]
    partly_burried_data = partly_burried_data[partly_burried_data['Buried - Partly'] > 0]
    partly_burried_data = round(partly_burried_data.groupby('Season').mean(),1)
    partly_burried_data['Outcome'] = 'Buried - Partly'

    fully_burried_data = filtered_avy_size[['Season', 'Depth (ft)', 'Width (ft)', 'Vertical (ft)', 'Buried - Fully']]
    fully_burried_data = fully_burried_data[fully_burried_data['Buried - Fully'] > 0]
    fully_burried_data = round(fully_burried_data.groupby('Season').mean(),1)
    fully_burried_data['Outcome'] = 'Buried - Fully'

    # Append DFs into one 
    full_dataset = pd.concat([caught_data, carried_data, partly_burried_data, fully_burried_data])
    full_dataset = full_dataset.drop(columns = ['Caught', 'Carried', 'Buried - Partly', 'Buried - Fully'])
    full_dataset = full_dataset.reset_index()
    melted_df = pd.melt(
        full_dataset,
        id_vars=['Season', 'Outcome'],
        value_vars=['Depth (ft)', 'Width (ft)', 'Vertical (ft)'],
        var_name='Dimensions',
        value_name='Values'
    )

    # Chart data
    outcome_chart = (
        alt.Chart(melted_df)
        .mark_bar()
        .encode(
            x=alt.X("Outcome:N", title=''),
            y=alt.Y("Values:Q", title=''),
            color=alt.Color('Dimensions:N', legend=None),
            xOffset="Dimensions:N" # Groups bars for each Subgroup
        )
    )
    st.altair_chart(outcome_chart, use_container_width=True)

col_graph_4, col_comments_5 = st.columns(2, gap='small', vertical_alignment='center')

with col_graph_4:
    st.markdown('#####')
    st.markdown('##### Avalanches by Month (all seasons)')

    all_avys = df.copy()

    # Create dataframe with desired data (Season, Dates)
    all_avys = all_avys[['Season', 'Date']]

    # Creat new date column with year removed
    all_avys['Month Num'] = all_avys['Date'].dt.month
    all_avys = pd.DataFrame(all_avys)
    month_map = {
        1.0: 'Jan',
        2.0: 'Feb',
        3.0: 'Mar',
        4.0: 'Apr',
        5.0: 'May',
        6.0: 'Jun',
        7.0: 'Jul',
        8.0: 'Aug',
        9.0: 'Sep',
        10.0: 'Oct',
        11.0: 'Nov',
        12.0: 'Dec'
    }
    all_avys['Month'] = all_avys['Month Num'].map(month_map)
    all_avys = all_avys.drop(columns = ['Date', 'Month Num'])

    # Groupby Season and Date with a new count column; reset_index applies column name
    all_avys = all_avys.groupby('Season').value_counts().reset_index()

    # Remove Unkown Seasons
    all_avys = all_avys[all_avys['Season'] != 'Unknown']

    # Chart data with altair to make interactive
    all_avys_chart = (alt.Chart(all_avys)
                      .mark_line(point=True)
                      .encode(
                          x=alt.X('Month:N', title=''),
                          y=alt.Y('count:Q', title=''),
                          color=alt.Color('Season:N', title='Season'),
                          tooltip=['Month:N', 'Season:N', 'Count:Q']
                      )
                      .interactive() # Enables zoom and pan
                      .properties(
                          width=700,
                          height=500
                      )
                    )
    st.altair_chart(all_avys_chart, use_container_width=True)

with col_comments_5:
    st.markdown('#####')
    st.markdown("**Observation**: The northeast aspect consistently experiences the highest number of avalanches. \
                \n\n- **Hypothesis**: This is due to consistent wind loading, with westerly winds depositing snow on east-facing slopes. \
                Additionally, being in the Northern Hemisphere, Utah's north-facing slopes receive less sunlight, leading to colder temperatures and increased faceting.")
    st.markdown("**Observation**: The largest weak layer in Utah is consistently caused by faceting, except for instances of a new snow/old snow interface (e.g., crusts, facets, or surface hoar) observed in 2019/2020 and 2016/2017.\
                \n\n- **Hypothesis**: The 2019/2020 and 2016/2017 seasons experienced record-breaking and consistent snowfall throughout.")
    st.markdown("**Observation**: Salt Lake has the most recorded week layers.\
                \n\n- **Hypothesis**: Salt Lake sees the most backcountry traffic due to its abundant snowfall and easy access from nearby large populations.")
    st.markdown("**Observation**: The data contradicts the initial assumption that larger avalanches are more likely to result in burial.\
                \n\n- **Hypothesis**: There is no strong correlation between avalanche size and the likelihood of full burial.")
    st.markdown("**Observation**: February 2021 recorded the highest number of avalanches.\
                \n\n- **Hypothesis**: This was due to record snowfall over a short period on a faceted weak layer.")
