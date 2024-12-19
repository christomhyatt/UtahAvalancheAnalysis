import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the avalanche dataset.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Select relevant columns
    columns_to_keep = [
        'Date', 'Region', 'Place', 'Trigger', 'Trigger: additional info',
        'Weak Layer', 'Depth', 'Width', 'Vertical', 'Aspect', 'Elevation',
        'Coordinates', 'Caught', 'Carried', 'Buried - Partly', 'Buried - Fully',
        'Injured', 'Killed'
    ]
    df = df[columns_to_keep]
    
    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def create_season_mapping():
    """
    Create a dictionary of winter seasons.
    
    Returns:
        dict: Mapping of season names to date ranges
    """
    return {
        f"{year}/{year+1}": (
            pd.to_datetime(f"{year}-09-01"), 
            pd.to_datetime(f"{year+1}-07-01")
        ) for year in range(2010, 2025)
    }

def assign_winter_season(df, seasons):
    """
    Assign winter season to each record.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        seasons (dict): Season mapping
    
    Returns:
        pd.DataFrame: DataFrame with Season column added
    """
    # Create season columns
    for season, (start_date, end_date) in seasons.items():
        df[season] = ((df['Date'] >= start_date) & (df['Date'] <= end_date)).apply(
            lambda x: season if x else "Unknown"
        )
    
    # Consolidate into single Season column
    df['Season'] = df[list(seasons.keys())].apply(
        lambda row: next((season for season in row if season != 'Unknown'), 'Unknown'), 
        axis=1
    )
    
    # Drop intermediate season columns
    df = df.drop(columns=list(seasons.keys()))
    
    return df

def process_elevation_data(df, year):
    """
    Process elevation data for visualization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        year (str): Selected season year
    
    Returns:
        dict: Processed elevation data for SVG visualization
    """
    # Create a copy of the DataFrame
    avy_elevation = df[['Elevation', 'Season']].copy()
    
    # Clean and convert Elevation column to numerical
    avy_elevation['Elevation'] = avy_elevation['Elevation'].replace(regex=[',', "'"], value='')
    avy_elevation['Elevation Detail'] = pd.to_numeric(avy_elevation['Elevation'], errors="coerce")
    
    # Define elevation levels
    def determine_elevation_level(elevation):
        if elevation > 9000:
            return ">9K"
        elif 8000 < elevation <= 9000:
            return ">8K and <9K"
        elif elevation <= 8000:
            return "<8K"
        else:
            return "Unknown"
    
    avy_elevation['Elevation Level'] = avy_elevation['Elevation Detail'].apply(determine_elevation_level)
    
    # Count avalanches by elevation level for the selected year
    elevation_counts = avy_elevation[avy_elevation['Season'] == year]['Elevation Level'].value_counts()
    
    # Prepare data for SVG
    svg_data = {
        "top": {"range": ">9K", "count": elevation_counts.get(">9K", 0)},
        "middle": {"range": ">8K and <9K", "count": elevation_counts.get(">8K and <9K", 0)},
        "bottom": {"range": "<8K", "count": elevation_counts.get("<8K", 0)}
    }
    
    return svg_data

def create_elevation_svg(svg_data):
    """
    Create an SVG visualization of elevation data.
    
    Args:
        svg_data (dict): Processed elevation data
    
    Returns:
        str: SVG code as a string
    """
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
    return svg_code

def plot_weak_layer_distribution(df, year):
    """
    Create a bar plot of weak layer distribution.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        year (str): Selected season year
    
    Returns:
        plt.Figure: Matplotlib figure with weak layer distribution
    """
    # Filter data for the selected year
    weak_layer = df[df['Season'] == year]['Weak Layer']
    weak_layer_counts = weak_layer.value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    wl_bars = ax.bar(weak_layer_counts.index, weak_layer_counts.values, color='skyblue', edgecolor='black')

    # Add value labels on top of each bar
    for bar in wl_bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom')

    ax.set_title("Weak Layer Distribution", fontsize=16)
    ax.set_ylabel("Avalanches", fontsize=12)
    ax.set_xticklabels(weak_layer_counts.index, rotation=45, ha='right', fontsize=8)

    return fig

def process_avalanche_size_data(df, year):
    """
    Process avalanche size data for different outcomes.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        year (str): Selected season year
    
    Returns:
        pd.DataFrame: Processed avalanche size data
    """
    # Create a copy of the DataFrame
    avy_size = df[['Season', 'Depth', 'Width', 'Vertical', 'Caught', 'Carried', 'Buried - Partly', 'Buried - Fully']].copy()
    
    # Clean and convert numerical columns
    numeric_columns = ['Depth', 'Width', 'Vertical', 'Caught', 'Carried', 'Buried - Partly', 'Buried - Fully']
    for col in numeric_columns:
        avy_size.loc[:, col] = avy_size[col].replace(regex=["'"], value='')
        avy_size.loc[:, col] = round(pd.to_numeric(avy_size[col], errors="coerce"), 1)
    
    # Filter for the selected year and remove rows with NaN
    avy_size = avy_size[avy_size['Season'] == year].dropna(subset=['Depth', 'Width', 'Vertical'])
    
    # Process different outcome categories
    outcome_categories = [
        ('Caught', 'Caught'),
        ('Carried', 'Carried'),
        ('Buried - Partly', 'Buried - Partly'),
        ('Buried - Fully', 'Buried - Fully')
    ]
    
    full_dataset = []
    for label, column in outcome_categories:
        outcome_data = avy_size[avy_size[column] > 0]
        if not outcome_data.empty:
            avg_data = outcome_data.groupby('Season')[['Depth', 'Width', 'Vertical']].mean().round(1)
            avg_data['Outcome'] = label
            full_dataset.append(avg_data)
    
    return pd.concat(full_dataset) if full_dataset else pd.DataFrame()

def plot_avalanche_outcomes(full_dataset):
    """
    Create a bar plot of avalanche outcomes by sizes.
    
    Args:
        full_dataset (pd.DataFrame): Processed avalanche outcomes data
    
    Returns:
        plt.Figure: Matplotlib figure with avalanche outcomes
    """
    # Prepare chart data
    outcomes = full_dataset['Outcome'].values
    metrics = ['Depth', 'Width', 'Vertical']
    values = full_dataset[metrics].T

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 5))
    x = range(len(outcomes))
    width = 0.2

    for i, metric in enumerate(metrics):
        ax.bar(
            [pos + i * width for pos in x],
            values.iloc[i],
            width=width,
            label=metric,
        )

    ax.set_title("Avalanche Outcomes by Sizes")
    ax.set_xlabel("Outcomes")
    ax.set_ylabel("Inches")
    ax.set_xticks([pos + width for pos in x])
    ax.set_xticklabels(outcomes)
    ax.legend(title="Metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return fig

def process_aspect_elevation_data(df, year):
    # Filter data on season selected
    yearly_data = df[df['Season'] == year]
    
    # Clean and convert Elevation column to numerical
    yearly_data['Elevation'] = yearly_data['Elevation'].replace(regex=[',', "'"], value='')
    yearly_data['Elevation Detail'] = pd.to_numeric(yearly_data['Elevation'], errors="coerce")

    # Define elevation levels
    def determine_elevation_level(elevation):
        if elevation > 9000:
            return ">9K"
        elif 8000 < elevation <= 9000:
            return ">8K and <9K"
        elif elevation <= 8000:
            return "<8K"
        else:
            return "Unknown"
    
    yearly_data['Elevation Level'] = yearly_data['Elevation Detail'].apply(determine_elevation_level)

    def standardize_aspect(aspect):
        """"""
        aspect = str(aspect).strip().upper()

        aspect_map = {
            "N": "North",
            "NE": "Northeast",
            'E': 'East', 
            'SE': 'Southeast', 
            'S': 'South', 
            'SW': 'Southwest', 
            'W': 'West', 
            'NW': 'Northwest'
        }

        return aspect_map.get(aspect, aspect)
    
    yearly_data['Aspect_Standardized'] = yearly_data['Aspect'].apply(standardize_aspect)

    return yearly_data

def create_wind_rose_diagram(df):
    grouped_data = df.groupby(['Aspect_Standardized', 'Elevation Level']).size().reset_index(name='Count')

    # Define color map for figure
    color_map = {
        '<8K': '#66c2a5',   # Green
        '<8K and <9K': '#fc8d62',  # Orange
        '>9K': '#8da0cb', # Blue
        'Unknown': '#e5e5e5'     # Light Gray
    }

    # Prepare data for windrose
    aspects = ['North', 'Northeast', 'East', 'Southeast', 'South', 'Southwest', 'West', 'Northwest']
    
    # Initialize figure
    fig = go.Figure()

    # Elevation categories to plot
    elevation_categories = ['<8K', '<8K and <9K', '>9K']

    # Plot each elevation category
    for category in elevation_categories:
        category_data = grouped_data[grouped_data['Elevation Level'] == category]

        # Ensure all aspects are represented
        full_data = pd.DataFrame({
            'Aspect_Standardized': aspects,
            'Count': [category_data[category_data['Aspect_Standardized'] == asp]['Count'].sum() if asp in category_data['Aspect_Standardized'].values else 0 for asp in aspects]
        })

        # Add trace
        fig.add_trace(go.Barpolar(
            r=full_data['Count'],
            theta=aspects,
            name=category,
            marker_color=color_map[category],
            opacity=0.8
        ))

    # Customize layout
    fig.update_layout(
        title='Avalanche Distribution by Aspect and Elevation',
        polar=dict(
            radialaxis=dict(visible=True, ticksuffix=' avalanches'),
            angularaxis=dict(direction="clockwise")
        ),
        legend_title_text='Elevation Categories',
        height=600,
        width=800
    )
    
    return fig

def add_wind_rose_to_dashboard(df, year):
    # Process data
    processed_data = process_aspect_elevation_data(df, year)
    
    # Create wind rose diagram
    wind_rose_fig = create_wind_rose_diagram(processed_data)
    
    # Display in Streamlit
    st.plotly_chart(wind_rose_fig, use_container_width=True)


def configure_streamlit_page():
    """
    Configure Streamlit page settings.
    """
    st.set_page_config(
        layout="wide",
        page_icon="🏔️",
        menu_items={
            'Get help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )

def main():
    """
    Main function to run the Streamlit dashboard.
    """
    # Configure page
    configure_streamlit_page()

    # Define filepath and seasons
    filepath = '/Users/chrishyatt/Library/Mobile Documents/com~apple~CloudDocs/Projects/gh_repos/AvyDash/avalanches.csv'
    seasons = create_season_mapping()

    # Load and preprocess data
    df = load_and_preprocess_data(filepath)
    df = assign_winter_season(df, seasons)

    # Page layout
    col_header_1, col_header_2 = st.columns([0.9, 0.2], gap="small", vertical_alignment="center")
    
    with col_header_2:
        year = st.selectbox('', sorted(seasons.keys(), reverse=True), placeholder="Select a season", label_visibility="collapsed", index=0)

    with col_header_1:
        st.title(f"_{year}_ Avalanches at a Glance")

    # Create columns for visualizations
    col1, col2 = st.columns(2, gap="small", vertical_alignment="center")

    with col1:
        # region = st.selectbox('', list(seasons.keys()), placeholder="Select a season", label_visibility="collapsed", index=0)

        # Elevation visualization
        svg_data = process_elevation_data(df, year)
        svg_code = create_elevation_svg(svg_data)
        st.markdown(f"""
        <div style="text-align: center;">
            {svg_code}
        """, unsafe_allow_html=True)

        # Weak layer distribution
        weak_layer_fig = plot_weak_layer_distribution(df, year)
        st.pyplot(weak_layer_fig)

    with col2:
        # Avalanche size and outcomes
        full_dataset = process_avalanche_size_data(df, year)
        if not full_dataset.empty:
            avalanche_outcomes_fig = plot_avalanche_outcomes(full_dataset)
            st.pyplot(avalanche_outcomes_fig)
        else:
            st.write("No avalanche outcome data available for the selected season.")

        st.subheader(f"Avalanche Aspect Dsitribution")
        add_wind_rose_to_dashboard(df, year)

if __name__ == "__main__":
    main()