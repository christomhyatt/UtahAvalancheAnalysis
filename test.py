import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import streamlit as st

# Example DataFrame
data = {
    "Aspect": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
    "Elevation": [10000, 9500, 9000, 8500, 8000, 7500, 7000, 6500],
    "Avalanche Count": [5, 15, 20, 10, 3, 8, 2, 12]
}
df = pd.DataFrame(data)

# Map Aspect to Angles (N=0, E=90, etc.)
aspect_to_angle = {
    "N": 0, "NE": 45, "E": 90, "SE": 135,
    "S": 180, "SW": 225, "W": 270, "NW": 315
}
df["Angle"] = df["Aspect"].map(aspect_to_angle)

# Normalize Elevation for radial scaling
elevation_min, elevation_max = df["Elevation"].min(), df["Elevation"].max()
df["Normalized Elevation"] = (df["Elevation"] - elevation_min) / (elevation_max - elevation_min)

# Plot Heatmap on Polar Coordinates
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

# Scatter plot with color-mapped Avalanche Count
cmap = plt.cm.hot
norm = Normalize(vmin=df["Avalanche Count"].min(), vmax=df["Avalanche Count"].max())
colors = cmap(norm(df["Avalanche Count"]))

scatter = ax.scatter(
    np.radians(df["Angle"]),  # Convert angles to radians
    df["Normalized Elevation"],  # Radial distance
    c=colors,
    s=200,  # Marker size
    cmap=cmap,
    alpha=0.8
)

# Add colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.1)
cbar.set_label("Avalanche Count")

# Customize the plot
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xticks(np.radians(list(aspect_to_angle.values())))
ax.set_xticklabels(list(aspect_to_angle.keys()))
ax.set_yticklabels([f"{int(elevation_min + (e * (elevation_max - elevation_min)))} ft" for e in ax.get_yticks()])

# Integrate with Streamlit
st.pyplot(fig)
