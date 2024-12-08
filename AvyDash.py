import streamlit as st
import pandas as pd
import plotly.graph_objects as go


filepath = '/Users/chrishyatt/Library/Mobile Documents/com~apple~CloudDocs/Projects/gh_repos/AvyDash/UAC Avys 23-24.csv'
avy_csv_raw = pd.read_csv(filepath)
df = pd.DataFrame(avy_csv_raw)

df = df[['Date', 'Region', 'Place', 'Trigger', 'Trigger: additional info',
       'Weak Layer', 'Depth', 'Width', 'Vertical', 'Aspect', 'Elevation',
       'Coordinates', 'Caught', 'Carried', 'Buried - Partly', 'Buried - Fully',
       'Injured', 'Killed']]

# print(df['Accident and Rescue Summary'].value_counts)
# print(df['Terrain Summary'].value_counts)
# print(df['Weather Conditions and History'].value_counts)
st.set_page_config(
    layout="wide",
    page_icon="V",
    menu_items={
        'Get help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
        })

# title and filter page formatting
col_header_1, col_header_2 = st.columns([.9,.2], gap="small", vertical_alignment="center",)
with col_header_2:  
    year = st.selectbox('',('2023/24','2024/25'), placeholder="Select a season", label_visibility="collapsed", )
with col_header_1:
    st.title(f"_{year}_ Avalanches at a Glance")

# dashboards
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
with col1:
    # Create a custom-shaped mountain outline
    fig = go.Figure()

    # Top of the mountain (9k+)
    fig.add_shape(
        type="path",
        path="M 0, 10 L 3, 8 L 0, 6 Z", # Replace with the mountain's outline
        fillcolor="red",
        line_color="black",
        name="9k+ (190 avalanches)"
    )

    # Middle of the mountain (8-9k)
    fig.add_shape(
        type="path",
        path="M 0, 6 L 3, 8 L 3, 4 L 0, 6 Z",
        fillcolor="orange",
        line_color="black",
        name="8-9k (126 avalanches)"
    )

    # Bottom of the mountain (<8k)
    fig.add_shape(
        type="path",
        path="M 0, 4 L 3, 4 L 3, 0 L 0, 0 Z",
        fillcolor="green",
        line_color="black",
        name="<8k (92 avalanches)"
    )
    fig.add_shape()

    # Adjust layout
    fig.update_layout(
        title="Mountain Avalanche Data",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    st.plotly_chart(fig)

with col2:
    st.write("column 2 test")


