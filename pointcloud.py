#!/usr/bin/env python3
"""
Simple Raster to Point Cloud Converter
Converts a single raster to a point cloud for Cesium/TerriaMap visualization
"""

import numpy as np
import rasterio
import laspy
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_raster(raster_path):
    """Read raster data and metadata"""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        
        logger.info(f"Raster shape: {data.shape}")
        logger.info(f"Raster resolution: {src.res}")
        logger.info(f"Nodata value: {src.nodata}")
        
        metadata = {
            'transform': src.transform,
            'crs': src.crs,
            'bounds': src.bounds,
            'res': src.res,
            'nodata': src.nodata,
            'width': src.width,
            'height': src.height,
            'shape': data.shape
        }
        
        return data, metadata

def generate_point_cloud(raster_data, metadata, config, value_min, value_max):
    """Generate point cloud with density falloff, biased height, and variable randomness"""
    nodata = metadata['nodata']
    
    points = []
    valid_cells = 0
    skipped_cells = 0
    
    cell_width = metadata['res'][0]
    cell_height = metadata['res'][1]
    
    value_range = value_max - value_min
    if value_range == 0:
        value_range = 1

    for row in range(0, raster_data.shape[0], config['downsample_factor']):
        for col in range(0, raster_data.shape[1], config['downsample_factor']):
            value = raster_data[row, col]
            
            if nodata is not None and value == nodata:
                skipped_cells += 1
                continue
            
            if config['min_value_threshold'] and value < config['min_value_threshold']:
                skipped_cells += 1
                continue
            
            valid_cells += 1

            normalized_value = (value - value_min) / value_range
            probability = (1 - normalized_value) ** config['density_falloff_factor']
            
            x_center = col * cell_width
            y_center = (raster_data.shape[0] - row) * cell_height
            
            for _ in range(config['points_per_cell']):
                if np.random.random() < probability:
                    x_offset = (np.random.random() - 0.5) * cell_width * config['horizontal_randomness']
                    y_offset = (np.random.random() - 0.5) * cell_height * config['horizontal_randomness']
                    x = x_center + x_offset
                    y = y_center + y_offset
                    
                    biased_base_height = (normalized_value ** config['height_bias_factor']) * value_max
                    z = biased_base_height * config['vertical_scale']
                    
                    if config['add_vertical_randomness']:
                        # DYNAMIC RANDOMNESS CALCULATION
                        randomness_range = config['max_vertical_randomness'] - config['min_vertical_randomness']
                        current_randomness = config['min_vertical_randomness'] + (normalized_value * randomness_range)
                        z += np.random.random() * current_randomness
                    
                    z = min(z, config['max_height'])
                    points.append([x, y, z, value])
    
    logger.info(f"Processed {valid_cells} valid cells, skipped {skipped_cells} cells")
    logger.info(f"Generated {len(points)} total points with density falloff")
    logger.info(f"Local coordinate system: X[0, {raster_data.shape[1] * cell_width:.1f}], Y[0, {raster_data.shape[0] * cell_height:.1f}]")
    
    return np.array(points)

def create_las_file(points, output_path, metadata, value_name="value"):
    """Create LAS file from points - using local coordinates starting at 0,0"""
    if len(points) == 0:
        logger.error("No points to write!")
        return
    
    x, y, z, values = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    
    logger.info("Using local coordinate system (meters)")
    logger.info(f"X range: {np.min(x):.2f} to {np.max(x):.2f} meters")
    logger.info(f"Y range: {np.min(y):.2f} to {np.max(y):.2f} meters")
    
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [np.min(x), np.min(y), np.min(z)]
    
    header.add_extra_dim(laspy.ExtraBytesParams(
        name=value_name, 
        type=np.float32,
        description=f"{value_name} from raster"
    ))
    
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    setattr(las, value_name, values.astype(np.float32))
    
    if np.max(values) > np.min(values):
        intensity_normalized = ((values - np.min(values)) / (np.max(values) - np.min(values)) * 65535)
    else:
        intensity_normalized = np.zeros_like(values) * 32767
    las.intensity = intensity_normalized.astype(np.uint16)
    
    def hex_to_rgb_scaled(hex_color):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return tuple(c * 257 for c in rgb)

    color_hex_palette = ['#50f0e6', '#50ccaa', '#f0e641', '#ff5050', '#960032', '#7d2181']
    rgb_palette = [hex_to_rgb_scaled(c) for c in color_hex_palette]
    num_colors = len(rgb_palette)

    colors = np.zeros((len(points), 3), dtype=np.uint16)
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values)) if np.max(values) > np.min(values) else np.zeros_like(values)
    
    for i, norm_val in enumerate(normalized_values):
        if norm_val >= 1.0:
            colors[i] = rgb_palette[-1]
            continue
        
        scaled_val = norm_val * (num_colors - 1)
        idx1 = int(scaled_val)
        idx2 = idx1 + 1
        fraction = scaled_val - idx1
        
        start_color, end_color = rgb_palette[idx1], rgb_palette[idx2]
        
        r = int(start_color[0] * (1 - fraction) + end_color[0] * fraction)
        g = int(start_color[1] * (1 - fraction) + end_color[1] * fraction)
        b = int(start_color[2] * (1 - fraction) + end_color[2] * fraction)
        
        colors[i] = [r, g, b]

    las.red, las.green, las.blue = colors[:, 0], colors[:, 1], colors[:, 2]

    las.write(output_path)
    
    logger.info(f"Created LAS file: {output_path}")
    logger.info(f"Points: {len(points)}, Value range: {np.min(values):.3f} - {np.max(values):.3f}")
    
    meta_path = Path(output_path).with_suffix('.json')
    metadata_out = {
        'point_count': len(points),
        'value_name': value_name,
        'value_range': [float(np.min(values)), float(np.max(values))],
        'bounds': {
            'min': [float(np.min(x)), float(np.min(y)), float(np.min(z))],
            'max': [float(np.max(x)), float(np.max(y)), float(np.max(z))]
        },
        'coordinate_system': 'local',
        'units': 'meters',
        'origin': 'bottom-left corner at (0,0)',
        'raster_shape': metadata.get('shape', 'unknown')
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata_out, f, indent=2)
    
    logger.info(f"Saved metadata to: {meta_path}")
    
def create_terria_config(value_name, value_min, value_max, ion_asset_id="YOUR_ASSET_ID", ion_token="YOUR_TOKEN"):
    # ... (function remains the same)
    pass

if __name__ == "__main__":
    raster_path = r"C:\Users\Austin\Documents\DATABANK\Masters\Thesis\Code\final_method\lr_output\LUR_MAP_NO2.tif"  # Input raster file
    output_dir = r"C:\Users\Austin\Documents\DATABANK\Masters\Thesis\Code\final_method\lr_output"                 # Output directory
    value_name = "NO2" 
    
    # ========== YOUR FINAL CONFIGURATION ==========
    config = {
    # Start with a high max number of points, as falloff will reduce the total.
    'points_per_cell': 3,

    # General scaling for the base height from the raster value.
    'vertical_scale': 0.1,

    # --- Dynamic Vertical Randomness ---
    'add_vertical_randomness': True,
    # Randomness for points with the LOWEST raster values will be up to 300m.
    'min_vertical_randomness': 100.0,
    # Randomness for points with the HIGHEST raster values will be up to 1000m.
    'max_vertical_randomness': 1000,

    # How much points spread horizontally within a cell.
    'horizontal_randomness': 0.8,

    # --- Filtering and Caps ---
    'min_value_threshold': None, # Optional: ignore cells below this value.
    # A safety cap; set higher than your max possible height.
    'max_height': 1000,
    'downsample_factor': 1, # Process every cell.

    # --- Density and Height Biasing ---
    # A value > 2.0 creates a sharp drop in point density at high altitudes.
    'density_falloff_factor': 3,
    # A value of 1.0 means no height bias (linear scaling).
    'height_bias_factor': 1.0
    }
    # ============================================
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Reading raster: {raster_path}")
        raster_data, metadata = read_raster(raster_path)
        
        if metadata['nodata'] is not None:
            valid_data = raster_data[raster_data != metadata['nodata']]
        else:
            valid_data = raster_data
        value_min = float(valid_data.min())
        value_max = float(valid_data.max())
        logger.info(f"Value range for scaling: {value_min:.2f} to {value_max:.2f}")
        
        logger.info("Generating point cloud...")
        points = generate_point_cloud(raster_data, metadata, config, value_min, value_max)
        
        if len(points) == 0:
            logger.error("No valid points generated!")
            exit(1)
        
        output_las = Path(output_dir) / "point_cloud_final3.las"
        create_las_file(points, str(output_las), metadata, value_name)
        
        # ... (rest of the main block remains the same)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise