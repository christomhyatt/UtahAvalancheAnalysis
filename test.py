import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Function to create the heatmap
def create_avalanche_heatmap(df):
    # Create elevation bins
    elevation_bins = [5100, 6480, 7860, 9240, 10620, 12000]
    elevation_labels = ['5100-6480', '6480-7860', '7860-9240', '9240-10620', '10620-12000']
    
    # Add elevation category to the DataFrame
    df['elevation_category'] = pd.cut(df['elevation'], 
                                    bins=elevation_bins, 
                                    labels=elevation_labels)
    
    # Create a pivot table
    heatmap_data = pd.pivot_table(df,
                                 values='count',
                                 index='direction',
                                 columns='elevation_category',
                                 aggfunc='sum',
                                 fill_value=0)
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the heatmap
    sns.heatmap(heatmap_data,
                cmap='YlOrRd',
                annot=True,
                fmt='g',
                cbar_kws={'label': 'Number of Avalanches'},
                square=True,
                ax=ax)
    
    plt.title('Avalanche Frequency by Direction and Elevation')
    plt.xlabel('Elevation Range (ft)')
    plt.ylabel('Direction')
    plt.xticks(rotation=45)
    
    return fig

# Create sample data function (for testing)
def create_sample_data():
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    data = []
    
    for direction in directions:
        for elevation in np.linspace(5100, 12000, 100):
            count = np.random.poisson(5)
            data.append({
                'direction': direction,
                'elevation': elevation,
                'count': count
            })
    
    return pd.DataFrame(data)

# Streamlit app code
def main():
    st.title('Avalanche Analysis')
    
    # Create sample data (replace this with your actual data)
    df = create_sample_data()
    
    # Create the heatmap
    fig = create_avalanche_heatmap(df)
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Optional: Display the raw data
    if st.checkbox('Show raw data'):
        st.write(df)

if __name__ == "__main__":
    main()