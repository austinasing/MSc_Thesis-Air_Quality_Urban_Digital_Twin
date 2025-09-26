# create_titiler_app.py

from fastapi import FastAPI
from titiler.core.factory import TilerFactory
from titiler.core.dependencies import create_colormap_dependency
from rio_tiler.colormap import cmap as default_cmap
from titiler.extensions import wmsExtension
from titiler.core.errors import add_exception_handlers, DEFAULT_STATUS_CODES
import matplotlib.colors as mcolors
import numpy as np

# --- Colormap generation ---
COLORS = ['#50f0e6','#50ccaa','#f0e641','#ff5050','#960032', '#7d2181']
EAQI_LIMITS = {
    'no2': [0, 40, 90, 120, 230, 340, 1000],
    'o3': [0, 50, 100, 130, 240, 380, 800],
    'pm10': [0, 20, 40, 50, 100, 150, 1200],
    'pm25': [0, 10, 20, 25, 50, 75, 800],
    'so2': [0, 100, 200, 350, 500, 750, 1250],
    'eaqi': [0,1,2,3,4,5,6]
}
def create_linear_colormap(limits):
    """
    Creates a 256-entry linear colormap (keys 0-255) with a non-linear color
    distribution based on the provided pollutant value limits.
    """
    min_val, max_val = limits[0], limits[-1]

    # Normalize the pollutant value break points to the 0.0-1.0 range
    normalized_limits = [(l - min_val) / (max_val - min_val) for l in limits]

    # Create tuples of (normalized_value, color) for matplotlib
    color_stops = list(zip(normalized_limits[:-1], COLORS))
    color_stops.append((1.0, COLORS[-1]))

    # Create a continuous, non-linear colormap object
    cmap_obj = mcolors.LinearSegmentedColormap.from_list("custom_non_linear", color_stops)

    # Sample this continuous map at 256 linear intervals (from 0.0 to 1.0).
    # This "bakes in" the non-linear distribution into a standard linear map.
    colors_rgba = (cmap_obj(np.linspace(0, 1, 256)) * 255).astype(int)

    # Create the final dictionary with integer keys 0 through 255
    return {idx: rgba.tolist() for idx, rgba in enumerate(colors_rgba)}

CUSTOM_COLORMAPS = {
    pollutant: create_linear_colormap(limits)
    for pollutant, limits in EAQI_LIMITS.items()
}
# Register new colormaps
cmap = default_cmap.register(CUSTOM_COLORMAPS)
CustomColorMapParams = create_colormap_dependency(cmap)


# Create the FastAPI application instance
app = FastAPI(title="My Titiler App")

# Create an instance of TilerFactory.
# This will provide the COG endpoints we need.
cog = TilerFactory(
    extensions=[
        wmsExtension(),  # This adds the /wms endpoint
    ],
    colormap_dependency=CustomColorMapParams
)

# Register the factory's router.
# We will still mount it at the "/bands" prefix to match our proxy's expectations.
app.include_router(cog.router, prefix="/bands", tags=["Cloud Optimized GeoTIFF & WMS"])

# Add Titiler's default exception handlers
add_exception_handlers(app, DEFAULT_STATUS_CODES)