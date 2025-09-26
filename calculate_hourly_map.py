# Standalone script to calculate the final 3-step hourly map for a given timestamp.
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio import CRS
import requests
import joblib
import json
import os
import argparse
from datetime import datetime, timezone
from scipy.spatial import cKDTree
import warnings
import psycopg2
from psycopg2 import pool
import pyproj
# UPDATED: Added Resampling for COG overviews
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as RasterioResampling

os.environ["GTIFF_SRS_SOURCE"] = "EPSG"
os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
wkt_7801 = """PROJCS["BGS2005 / CCS2005",
GEOGCS["BGS2005",
DATUM["Bulgaria_Geodetic_System_2005",
    SPHEROID["GRS 1980",6378137,298.257222101],
    TOWGS84[0,0,0,0,0,0,0]],
PRIMEM["Greenwich",0,
    AUTHORITY["EPSG","8901"]],
UNIT["degree",0.0174532925199433,
    AUTHORITY["EPSG","9122"]],
AUTHORITY["EPSG","7798"]],
PROJECTION["Lambert_Conformal_Conic_2SP"],
PARAMETER["latitude_of_origin",42.6678756833333],
PARAMETER["central_meridian",25.5],
PARAMETER["standard_parallel_1",42],
PARAMETER["standard_parallel_2",43.3333333333333],
PARAMETER["false_easting",500000],
PARAMETER["false_northing",4725824.3591],
UNIT["metre",1,
AUTHORITY["EPSG","9001"]],
AUTHORITY["EPSG","7801"]]"""
wkt_4326 = """GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6236"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.0174532925199433,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""
project_crs = CRS.from_wkt(wkt_7801)
new_crs = CRS.from_wkt(wkt_4326)

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
# --- File & Path Configuration ---
BASE_DIR = r"C:\Users\Austin\Documents\DATABANK\Masters\Thesis\Code\final_prep"
MODEL_DIR = os.path.join(BASE_DIR, 'stage2', 'models')
LUR_MAP_DIR = os.path.join(BASE_DIR, 'stage1', 'maps')
STATION_INFO_WGS84_PATH = r"C:\Users\Austin\Documents\DATABANK\Masters\Thesis\Code\final_method\station_info3_wgs84.csv"
STATION_INFO_EPSG7801_PATH = r"C:\Users\Austin\Documents\DATABANK\Masters\Thesis\Code\final_method\station_info3.csv"
OUTPUT_RASTER_PATH = r"C:\Users\Austin\Documents\DATABANK\Masters\Thesis\Spatial_Data\master_raster6.tif"
CALCULATION_RASTER_PATH = r"C:\Users\Austin\Documents\DATABANK\Masters\Thesis\Code\final_prep\stage1\maps\LUR_MAP_NO2.tif"
OUTPUT_DIR = os.path.join(BASE_DIR, 'hourly_maps')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Database Configuration ---
DB_CONFIG = {
    "host": "localhost", "port": 5432, "dbname": "platform_db",
    "user": "postgres", "password": "postgres"
}

# --- Model & Pollutant Configuration ---
POLLUTANTS = ['NO2', 'O3', 'PM10', 'PM25', 'SO2']
LOG_CONSTANT = 1e-9
IDW_POWER = 2
IDW_NEIGHBORS = 7
IDW_MAX_DISTANCE = 150000

# --- EAQI Configuration ---
EAQI_THRESHOLDS = {
    'no2':   [0, 10, 25, 60, 100, 150],
    'o3':    [0, 60, 100, 120, 160, 180],
    'pm10':  [0, 15, 45, 120, 195, 270],
    'pm2.5': [0, 5, 15, 50, 90, 140],
    'so2':   [0, 20, 40, 125, 190, 275]
}
# NEW: Added limits for rescaling, from orchestrator.py
EAQI_LIMITS = {
    'no2': [0, 40, 90, 120, 230, 340, 1000],
    'o3': [0, 50, 100, 130, 240, 380, 800],
    'pm10': [0, 20, 40, 50, 100, 150, 1200],
    'pm25': [0, 10, 20, 25, 50, 75, 800],
    'so2': [0, 100, 200, 350, 500, 750, 1250],
    'eaqi': [0, 1, 2, 3, 4, 5, 6]
}


# --- Global Mapping Variables ---
calc_raster_meta = {}
calc_master_mask = None
calc_grid_coords = None
output_raster_meta = {}


# ==============================================================================
# 1. HELPER & INITIALIZATION FUNCTIONS
# ==============================================================================
db_pool = None

def initialize_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = psycopg2.pool.SimpleConnectionPool(1, 5, **DB_CONFIG)

def get_db_connection():
    if db_pool is None:
        initialize_db_pool()
    return db_pool.getconn()

def return_db_connection(conn):
    if db_pool and conn:
        db_pool.putconn(conn)

def initialize_calculation_raster(file_path):
    """Initializes global mapping variables for the CALCULATION grid (EPSG:7801)."""
    global calc_raster_meta, calc_master_mask, calc_grid_coords
    print(f"Initializing CALCULATION raster (EPSG:7801) from: {file_path}")
    with rasterio.open(file_path) as src:
        array = src.read(1)
        shape = src.shape
        nodata_val = src.nodata
        calc_master_mask = (array == nodata_val)
        valid_indices = np.where(~calc_master_mask)
        rows, cols = valid_indices
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        calc_grid_coords = np.vstack((xs, ys)).T
        calc_raster_meta = src.profile
        calc_raster_meta.update({'driver': 'GTiff', 'dtype': 'float32', 'compress': 'LZW', 'shape':shape})

def initialize_output_raster(file_path):
    """Initializes the metadata for the OUTPUT raster (WGS 84)."""
    global output_raster_meta
    print(f"Initializing OUTPUT raster metadata (WGS 84) from: {file_path}")
    with rasterio.open(file_path) as src:
        output_raster_meta = src.profile
        output_raster_meta.update({'driver': 'GTiff', 'dtype': 'float32', 'compress': 'LZW'})


def load_models_and_data(pollutant):
    print(f"\n--- Loading models and data for {pollutant} ---")
    try:
        lur_results_path = os.path.join(LUR_MAP_DIR, f'LUR_MODEL_RESULTS_{pollutant}.json')
        with open(lur_results_path, 'r') as f:
            step1_model_data = json.load(f)
        
        step1_map_path = os.path.join(LUR_MAP_DIR, f'LUR_MAP_{pollutant}.tif')
        with rasterio.open(step1_map_path) as src:
            step1_map = src.read(1)
            if step1_map.shape != calc_raster_meta['shape']:
                print(f"🔴 FATAL ERROR: Shape mismatch for {pollutant}. Step 1 map is {step1_map.shape} but calculation raster is {calc_raster_meta['shape']}.")
                return None, None, None, None

        step2_model_path = os.path.join(MODEL_DIR, f'step2_model_{pollutant}.joblib')
        step2_scaler_path = os.path.join(MODEL_DIR, f'step2_scaler_{pollutant}.joblib')
        step2_model = joblib.load(step2_model_path)
        step2_scaler = joblib.load(step2_scaler_path)

        print(f"Successfully loaded all components for {pollutant}.")
        return step1_model_data, step1_map, step2_model, step2_scaler
    except FileNotFoundError as e:
        print(f"🔴 Error: Could not find a model file for {pollutant}. {e}")
        return None, None, None, None

# ==============================================================================
# 2. DATA FETCHING & PREPARATION
# ==============================================================================
def fetch_meteo_data(station_info_df_wgs84, timestamp_dt):
    print("--- Fetching real-time meteorology data ---")
    url = "https://api.open-meteo.com/v1/forecast" 
    params = {
        "latitude": station_info_df_wgs84['latitude'].tolist(),
        "longitude": station_info_df_wgs84['longitude'].tolist(),
        "start_hour": timestamp_dt.strftime('%Y-%m-%dT%H:00'),
        "end_hour": timestamp_dt.strftime('%Y-%m-%dT%H:00'),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "timezone": "UTC", "wind_speed_unit": "ms"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status(); data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"🔴 Failed to fetch from Open-Meteo: {e}"); return None

    results = []
    for i, station_data in enumerate(data):
        station_info = station_info_df_wgs84.iloc[i]
        hourly_data = station_data['hourly']
        results.append({
            'station_id': station_info['station_id'],
            'temperature_c': hourly_data['temperature_2m'][0],
            'relative_humidity': hourly_data['relative_humidity_2m'][0],
            'wind_speed_ms': hourly_data['wind_speed_10m'][0],
            'wind_direction_deg': hourly_data['wind_direction_10m'][0]
        })
    meteo_df = pd.DataFrame(results)
    print("Successfully fetched and processed real-time meteorology data.")
    return meteo_df

def fetch_observed_pollutant_data(pollutant, timestamp_dt):
    conn = get_db_connection()
    query = """
    SELECT station_id, reading_value
    FROM sensor_data
    WHERE measurement = %s AND measured_time >= %s AND measured_time < (%s + INTERVAL '1 hour');
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, (pollutant, timestamp_dt, timestamp_dt))
            results = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
        df = pd.DataFrame(results, columns=colnames)
        return df
    finally:
        return_db_connection(conn)


# ==============================================================================
# 3. SPATIAL INTERPOLATION, REPROJECTION & SAVING
# ==============================================================================
def create_idw_array(station_locs, values, power=IDW_POWER, n_neighbors=IDW_NEIGHBORS, max_distance=IDW_MAX_DISTANCE):
    if calc_master_mask is None or calc_grid_coords is None:
        raise ValueError("Calculation raster not initialized.")

    tree = cKDTree(station_locs)
    k = min(n_neighbors, len(station_locs))
    distances, indices = tree.query(calc_grid_coords, k=k)

    if k == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    distances[distances > max_distance] = np.inf
    distances[distances < 1e-10] = 1e-10
    weights = 1.0 / (distances ** power)
    sum_of_weights = np.sum(weights, axis=1)
    values_reshaped = values[indices]
    weighted_values = np.sum(weights * values_reshaped, axis=1)

    interpolated_values = np.divide(
        weighted_values, sum_of_weights,
        out=np.zeros_like(weighted_values),
        where=sum_of_weights != 0
    )

    idw_array = np.full(calc_raster_meta['shape'], 0.0, dtype=np.float32)
    idw_array[~calc_master_mask] = interpolated_values
    idw_array[calc_master_mask] = calc_raster_meta['nodata']
    return idw_array

def reproject_array(source_array, src_meta, dst_crs=new_crs):
    """Reprojects a numpy array from the calculation CRS to the destination CRS."""
    print(f"Reprojecting final map from {src_meta['crs']} to {dst_crs}...")
    
    src_bounds = array_bounds(src_meta['height'], src_meta['width'], src_meta['transform'])
    src_crs = project_crs
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, src_meta['width'], src_meta['height'], *src_bounds)
    
    dst_meta = src_meta.copy()
    dst_meta.update({
        'crs': dst_crs, 'transform': dst_transform, 
        'width': dst_width, 'height': dst_height,
        'nodata': output_raster_meta.get('nodata', -9999)
    })
    
    destination_array = np.empty((dst_height, dst_width), dtype=source_array.dtype)
    
    reproject(
        source=source_array,
        destination=destination_array,
        src_transform=src_meta['transform'],
        src_crs=src_crs,
        src_nodata=src_meta['nodata'],
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=dst_meta['nodata'],
        resampling=Resampling.bilinear)
        
    return destination_array, dst_meta

# NEW: Function to rescale data to uint16 for web visualization
def rescale_to_uint16(array, measurement, meta):
    """Rescales float pollutant array to UInt16 for web visualization."""
    if measurement not in EAQI_LIMITS:
        print(f"Warning: No rescale limits found for {measurement}. Cannot rescale.")
        return array, meta

    limits = EAQI_LIMITS[measurement]
    min_val, max_val = limits[0], limits[-2]
    original_nodata = meta.get('nodata')
    
    nodata_mask = (array == original_nodata) if original_nodata is not None else np.zeros(array.shape, dtype=bool)

    array_copy = array.copy()
    array_copy[nodata_mask] = min_val 
    np.clip(array_copy, min_val, max_val, out=array_copy)

    # Scale from 1 to 65535, leaving 0 for nodata
    scaled_data = (((array_copy - min_val) / (max_val - min_val)) * 65534) + 1 if max_val > min_val else np.ones(array_copy.shape)
    uint16_data = scaled_data.astype('uint16')
    uint16_data[nodata_mask] = 0

    updated_meta = meta.copy()
    updated_meta['dtype'] = 'uint16'
    updated_meta['nodata'] = 0
    return uint16_data, updated_meta

# NEW: Function to save arrays as Cloud-Optimized GeoTIFFs
def save_as_cog(array, file_path, profile):
    """Saves a numpy array as a Cloud-Optimized GeoTIFF."""
    cog_profile = profile.copy()
    cog_profile.pop('shape', None)
    cog_profile.update({
        'driver': 'COG',
        'compress': 'LZW',
    })

    try:
        with rasterio.open(file_path, 'w', **cog_profile) as dst:
            dst.write(array, 1)
            # Use average resampling for overviews
            overviews = [2, 4, 8, 16]
            dst.build_overviews(overviews, RasterioResampling.average)
        print(f" Saved Cloud-Optimized GeoTIFF to {file_path}")
    except Exception as e:
        print(f"Error saving COG file {file_path}: {e}")

# ==============================================================================
# 4. MAP CALCULATION STEPS
# ==============================================================================
def calculate_step2_residual_map(step2_model, step2_scaler, timestamp_dt, meteo_maps):
    print("--- Calculating Step 2 Residual Map ---")
    
    hour_sin = np.sin(2 * np.pi * timestamp_dt.hour / 24.0)
    hour_cos = np.cos(2 * np.pi * timestamp_dt.hour / 24.0)
    dayofyear_sin = np.sin(2 * np.pi * timestamp_dt.timetuple().tm_yday / 365.25)
    dayofyear_cos = np.cos(2 * np.pi * timestamp_dt.timetuple().tm_yday / 365.25)
    weekend = 1 if timestamp_dt.weekday() >= 5 else 0

    wind_dir_sin = np.sin(2 * np.pi * meteo_maps['wind_direction_deg'] / 360.0)
    wind_dir_cos = np.cos(2 * np.pi * meteo_maps['wind_direction_deg'] / 360.0)

    continuous_predictors = ['hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos', 'temperature_c', 'relative_humidity', 'wind_speed_ms', 'wind_dir_sin', 'wind_dir_cos']
    binary_predictors = ['weekend']
    all_predictors = continuous_predictors + binary_predictors
    
    feature_maps = {
        'hour_sin': hour_sin, 'hour_cos': hour_cos,
        'dayofyear_sin': dayofyear_sin, 'dayofyear_cos': dayofyear_cos,
        'temperature_c': meteo_maps['temperature_c'],
        'relative_humidity': meteo_maps['relative_humidity'],
        'wind_speed_ms': meteo_maps['wind_speed_ms'],
        'wind_dir_sin': wind_dir_sin, 'wind_dir_cos': wind_dir_cos,
        'weekend': weekend
    }

    print("  Building feature set for Step 2 prediction:")
    flat_features_scaled = {}
    for col in all_predictors:
        if col in continuous_predictors:
            idx = step2_scaler.feature_names_in_.tolist().index(col)
            mean, scale = step2_scaler.mean_[idx], step2_scaler.scale_[idx]
            
            if isinstance(feature_maps[col], np.ndarray):
                print(f"    - Using '{col}' as a spatial map (IDW).")
                scaled_map = (feature_maps[col] - mean) / scale
                flat_features_scaled[col] = scaled_map[~calc_master_mask]
            else:
                print(f"    - Using '{col}' as a constant value across map.")
                flat_features_scaled[col] = np.full(calc_grid_coords.shape[0], (feature_maps[col] - mean) / scale)
        
        elif col in binary_predictors:
            print(f"    - Using '{col}' as a constant value across map.")
            flat_features_scaled[col] = np.full(calc_grid_coords.shape[0], feature_maps[col])

    X_predict_df = pd.DataFrame(flat_features_scaled)[all_predictors]
    predicted_residuals_flat = step2_model.predict(X_predict_df)
    
    step2_map = np.full(calc_raster_meta['shape'], calc_raster_meta['nodata'], dtype=np.float32)
    step2_map[~calc_master_mask] = predicted_residuals_flat
    
    print("Step 2 map calculated.")
    return step2_map

def calculate_step3_correction_map(pollutant, timestamp_dt, step1_map, step2_map, station_info_epsg7801):
    print("--- Calculating Step 3 Error Correction Map ---")
    observed_df = fetch_observed_pollutant_data(pollutant, timestamp_dt)
    if observed_df.empty or len(observed_df) < 4:
        print("Not enough observed data for Step 3 correction. Returning zero map.")
        zero_map = np.full(calc_raster_meta['shape'], 0.0, dtype=np.float32)
        zero_map[calc_master_mask] = calc_raster_meta['nodata']
        return zero_map

    preds_df = station_info_epsg7801.merge(observed_df, on='station_id')
    coords = [tuple(map(float, coord.strip('()').split(','))) for coord in preds_df['location']]
    row, col = rasterio.transform.rowcol(calc_raster_meta['transform'], [c[0] for c in coords], [c[1] for c in coords])
    preds_df['step1_pred'] = step1_map[row, col]
    preds_df['step2_resid'] = step2_map[row, col]
    preds_df['step2_pred'] = preds_df['step1_pred'] + preds_df['step2_resid']
    preds_df['step2_error'] = preds_df['reading_value'] - preds_df['step2_pred']
    
    print(f"\n--- Station Residuals for {pollutant} Step 3 Correction ---")
    print(preds_df[['station_id', 'reading_value', 'step2_pred', 'step2_error']].round(4).to_string())
    print("-" * 60)
    
    # REMOVED the over-correction prevention logic to ensure map matches observations
    
    station_locs = np.array(coords)
    errors = preds_df['step2_error'].values
    step3_map = create_idw_array(station_locs, errors)
    
    print("Step 3 map calculated.")
    return step3_map

def calculate_eaqi_map(pollutant_arrays: dict):
    print("\n--- Calculating EAQI Map (Continuous Interpolated Method) ---")
    
    sub_indices = {}
    for pollutant_key, array in pollutant_arrays.items():
        threshold_key = 'pm2.5' if pollutant_key == 'pm25' else pollutant_key
        
        if threshold_key in EAQI_THRESHOLDS:
            thresholds = EAQI_THRESHOLDS[threshold_key]
            levels = np.arange(1, len(thresholds) + 1)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                continuous_index = np.interp(array, thresholds, levels)
            
            sub_indices[threshold_key] = continuous_index
        else:
            print(f"Warning: Missing EAQI thresholds for {pollutant_key}. Skipping.")

    if not sub_indices:
        print("🔴 No pollutant maps available to calculate EAQI."); return None
        
    stacked_indices = np.stack(list(sub_indices.values()), axis=0)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        eaqi_map = np.nanmax(stacked_indices, axis=0)
        
    eaqi_map[calc_master_mask] = calc_raster_meta['nodata']

    print("EAQI map calculated.")
    return eaqi_map

# ==============================================================================
# 5. VERIFICATION
# ==============================================================================
def verify_station_values_pre_reprojection(pollutant, timestamp_dt, final_map_array, station_info_epsg7801_df):
    print(f"\n--- Verifying Map Values for {pollutant} (PRE-REPROJECTION @ EPSG:7801) ---")
    observed_df = fetch_observed_pollutant_data(pollutant, timestamp_dt)
    if observed_df.empty:
        print("No observed station data for this hour to verify against.")
        return

    verification_df = pd.merge(station_info_epsg7801_df, observed_df, on='station_id')
    if verification_df.empty: return

    coords = [tuple(map(float, coord.strip('()').split(','))) for coord in verification_df['location']]
    rows, cols = rasterio.transform.rowcol(calc_raster_meta['transform'], [c[0] for c in coords], [c[1] for c in coords])
    verification_df['map_value'] = final_map_array[rows, cols]

    print("┌" + "─" * 15 + "┬" + "─" * 20 + "┬" + "─" * 20 + "┐")
    print(f"│ {'Station ID':<13} │ {'Observed Value':<18} │ {'Map Value (Raw)':<18} │")
    print("├" + "─" * 15 + "┼" + "─" * 20 + "┼" + "─" * 20 + "┤")
    for _, row in verification_df.iterrows():
        print(f"│ {row['station_id']:<13} │ {row['reading_value']:<18.4f} │ {row['map_value']:<18.4f} │")
    print("└" + "─" * 15 + "┴" + "─" * 20 + "┴" + "─" * 20 + "┘")

def verify_station_values_post_reprojection(pollutant, timestamp_dt, station_info_wgs84_df):
    # UPDATED: This function now reads the COG and un-scales the uint16 data for verification
    print(f"\n--- Verifying Map Values for {pollutant} (POST-SAVE from COG) ---")
    
    pollutant_key = pollutant.lower()
    filename = f"{timestamp_dt.strftime('%Y%m%d%H%M')}_{pollutant_key}.tif"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"🔴 Verification skipped: File not found at {filepath}"); return

    observed_df = fetch_observed_pollutant_data(pollutant, timestamp_dt)
    if observed_df.empty: return

    verification_df = pd.merge(station_info_wgs84_df, observed_df, on='station_id')
    if verification_df.empty: return
        
    coords = list(zip(verification_df['longitude'], verification_df['latitude']))
    
    with rasterio.open(filepath) as src:
        sampled_values_uint16 = [val[0] for val in src.sample(coords)]
        verification_df['map_value_uint16'] = sampled_values_uint16

    # Un-scale the values
    limits = EAQI_LIMITS.get(pollutant_key)
    if not limits:
        print(f"Cannot un-scale {pollutant_key}, no limits found."); return
        
    min_val, max_val = limits[0], limits[-2]
    
    scaled_vals = verification_df['map_value_uint16'].values
    unscaled_vals = (((scaled_vals - 1) / 65534) * (max_val - min_val)) + min_val
    # Set to 0 where the scaled value was 0 (nodata)
    unscaled_vals[scaled_vals == 0] = 0.0
    verification_df['map_value_unscaled'] = unscaled_vals

    print("┌" + "─" * 15 + "┬" + "─" * 20 + "┬" + "─" * 20 + "┐")
    print(f"│ {'Station ID':<13} │ {'Observed Value':<18} │ {'Map Value (Final)':<18} │")
    print("├" + "─" * 15 + "┼" + "─" * 20 + "┼" + "─" * 20 + "┤")
    for _, row in verification_df.iterrows():
        print(f"│ {row['station_id']:<13} │ {row['reading_value']:<18.4f} │ {row['map_value_unscaled']:<18.4f} │")
    print("└" + "─" * 15 + "┴" + "─" * 20 + "┴" + "─" * 20 + "┘")


# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
def main(timestamp_str):
    try:
        timestamp_dt = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
    except ValueError:
        print("🔴 Invalid timestamp format. Please use ISO format (e.g., YYYY-MM-DDTHH:MM:SS)"); return

    initialize_calculation_raster(CALCULATION_RASTER_PATH)
    initialize_output_raster(OUTPUT_RASTER_PATH)
    initialize_db_pool()

    station_info_wgs84 = pd.read_csv(STATION_INFO_WGS84_PATH)
    station_info_epsg7801 = pd.read_csv(STATION_INFO_EPSG7801_PATH)
    
    meteo_df = fetch_meteo_data(station_info_wgs84, timestamp_dt)
    if meteo_df is None: return
    
    meteo_df_epsg7801 = station_info_epsg7801.merge(meteo_df, on='station_id')

    meteo_maps = {}
    meteo_vars = ['temperature_c', 'relative_humidity', 'wind_speed_ms', 'wind_direction_deg']
    station_locs_epsg7801 = np.array([tuple(map(float, coord.strip('()').split(','))) for coord in meteo_df_epsg7801['location']])

    for var in meteo_vars:
        print(f"Interpolating {var} map in EPSG:7801...")
        values = meteo_df_epsg7801[var].values
        meteo_maps[var] = create_idw_array(station_locs_epsg7801, values)

    final_pollutant_maps_calc_crs = {}
    for p in POLLUTANTS:
        step1_model_data, step1_map, step2_model, step2_scaler = load_models_and_data(p)
        if step1_map is None: continue

        step2_residual_map = calculate_step2_residual_map(step2_model, step2_scaler, timestamp_dt, meteo_maps)
        step3_correction_map = calculate_step3_correction_map(p, timestamp_dt, step1_map, step2_residual_map, station_info_epsg7801)

        final_map_calc_crs = step1_map + step2_residual_map + step3_correction_map
        final_map_calc_crs[final_map_calc_crs < 0] = 0
        final_map_calc_crs[calc_master_mask] = calc_raster_meta['nodata']
        final_pollutant_maps_calc_crs[p.lower()] = final_map_calc_crs

        #verify_station_values_pre_reprojection(p, timestamp_dt, final_map_calc_crs, station_info_epsg7801)
        
        final_map_wgs84, final_meta_wgs84 = reproject_array(final_map_calc_crs, calc_raster_meta)

        # UPDATED: Rescale and save as COG
        rescaled_array, rescaled_meta = rescale_to_uint16(final_map_wgs84, p.lower(), final_meta_wgs84)
        filename = f"{timestamp_dt.strftime('%Y%m%d%H%M')}_{p.lower()}.tif" # Filename updated
        filepath = os.path.join(OUTPUT_DIR, filename)
        save_as_cog(rescaled_array, filepath, rescaled_meta)

        #verify_station_values_post_reprojection(p, timestamp_dt, station_info_wgs84)

    if final_pollutant_maps_calc_crs:
        eaqi_map_calc_crs = calculate_eaqi_map(final_pollutant_maps_calc_crs)
        if eaqi_map_calc_crs is not None:
            eaqi_map_wgs84, eaqi_meta_wgs84 = reproject_array(eaqi_map_calc_crs, calc_raster_meta)

            # UPDATED: Rescale and save EAQI map as COG
            rescaled_eaqi, rescaled_eaqi_meta = rescale_to_uint16(eaqi_map_wgs84, 'eaqi', eaqi_meta_wgs84)
            filename = f"{timestamp_dt.strftime('%Y%m%d%H%M')}_eaqi.tif" # Filename updated
            filepath = os.path.join(OUTPUT_DIR, filename)
            save_as_cog(rescaled_eaqi, filepath, rescaled_eaqi_meta)
            
    if db_pool:
        db_pool.closeall()
    print("\nHourly mapping process finished.")


if __name__ == "__main__":
    # UPDATED: Added argument parsing to be callable from orchestrator
    parser = argparse.ArgumentParser(description="Generate 3-step air quality maps for a given timestamp.")
    parser.add_argument('--timestamp', type=str, required=True, help="Timestamp in ISO format (e.g., YYYY-MM-DDTHH:MM:SS)")
    args = parser.parse_args()
    main(args.timestamp)

