# This script orchestrates the data ingestion and hourly map generation processes.
import os
import logging
import subprocess
import sys
import json
import rasterio
import numpy as np
import psycopg2
from psycopg2 import pool
from datetime import datetime, timedelta, timezone, date
from tqdm import tqdm
from rasterio.io import MemoryFile
import warnings
import pyproj
os.environ["GTIFF_SRS_SOURCE"] = "EPSG"
os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()

# Import the main function from the data ingestion script
from data_ingestor import run_data_ingestion, initialize_db_pool as init_ingestion_db, db_pool as ingestion_db_pool

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# -- Logging Setup --
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -- Path Configuration --
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOURLY_MAP_SCRIPT_PATH = os.path.join(BASE_DIR, 'calculate_hourly_map.py')
HOURLY_MAPS_DIR = os.path.join(BASE_DIR, 'hourly_maps')
METADATA_PATH = os.path.join(HOURLY_MAPS_DIR, 'metadata_lookup.json')
MASTER_RASTER_PATH = r'C:\Users\Austin\Documents\DATABANK\Masters\Thesis\Spatial_Data\master_raster6.tif'
DAILY_MAP_STATE_FILE = os.path.join(BASE_DIR, 'daily_map_last_run.txt')
os.makedirs(HOURLY_MAPS_DIR, exist_ok=True)

# -- Database Configuration --
DB_CONFIG = {
    "host": "localhost", "port": 5432, "dbname": "platform_db",
    "user": "postgres", "password": "postgres"
}

# -- Map Generation Configuration --
# Maps pollutants for daily aggregation and provides their DB names
POLLUTANTS_MAPPING = {'o3': 'O3', 'no2': 'NO2', 'so2': 'SO2', 'pm10': 'PM10', 'pm25': 'PM25'}

# -- Global Mapping Variables --
raster_meta = {}
db_pool_orchestrator = None # Separate pool for this script's functions

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def initialize_orchestrator_db_pool():
    global db_pool_orchestrator
    if db_pool_orchestrator is None:
        db_pool_orchestrator = psycopg2.pool.SimpleConnectionPool(1, 5, **DB_CONFIG)

def get_orchestrator_db_conn():
    if db_pool_orchestrator is None: initialize_orchestrator_db_pool()
    return db_pool_orchestrator.getconn()

def return_orchestrator_db_conn(conn):
    if db_pool_orchestrator and conn: db_pool_orchestrator.putconn(conn)

def get_master_metadata(file_path):
    """Initializes global raster metadata from a template raster."""
    global raster_meta
    logging.info(f"Reading template raster from: {file_path}")
    with rasterio.open(file_path) as src:
        raster_meta = src.profile
    logging.info("Master raster metadata initialized successfully.")

def upload_raster_to_postgres(array, model_date, measurement, conn):
    """Uploads a numpy array as a raster to the daily_historical_maps table."""
    profile = raster_meta.copy()
    profile.pop('shape', None)
    profile['dtype'] = array.dtype

    try:
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(array, 1)
            raster_bytes = memfile.read()
    except Exception as e:
        logging.error(f"Error creating in-memory raster: {e}"); return

    sql = """
    INSERT INTO daily_historical_maps (modeled_date, measurement, raster_data)
    VALUES (%s, %s::map_type, ST_FromGDALRaster(%s))
    ON CONFLICT (modeled_date, measurement) DO UPDATE SET raster_data = EXCLUDED.raster_data;
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (model_date, measurement.upper(), raster_bytes))
            conn.commit()
    except psycopg2.Error as e:
        logging.error(f"DB error during daily raster upload for {measurement}: {e}"); conn.rollback()

# ==============================================================================
# 3. ORCHESTRATION LOGIC
# ==============================================================================

def run_hourly_mapping_orchestration():
    """
    Calls the external mapping script for the last 24 hours and manages the metadata file.
    """
    logging.info("--- Starting Hourly Mapping Orchestration Process ---")
    
    if not os.path.exists(HOURLY_MAP_SCRIPT_PATH):
        logging.error(f"Mapping script not found at: {HOURLY_MAP_SCRIPT_PATH}"); return

    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = []

    last_complete_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    required_timestamps = [last_complete_hour - timedelta(hours=i) for i in range(24)]
    required_ts_strs = {ts.strftime("%Y-%m-%d %H:%M:%S") for ts in required_timestamps}
    
    existing_timestamps_in_window = {item['timestep'] for item in metadata if item['timestep'] in required_ts_strs}
    new_metadata = [item for item in metadata if item['timestep'] in existing_timestamps_in_window]

    for timestamp in tqdm(required_timestamps, desc="Generating hourly maps"):
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        if ts_str in existing_timestamps_in_window:
            continue

        logging.info(f"Queueing map generation for timestamp: {ts_str}")
        command = [sys.executable, HOURLY_MAP_SCRIPT_PATH, "--timestamp", timestamp.isoformat()]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            map_files = {}
            for p in list(POLLUTANTS_MAPPING.keys()) + ['eaqi']:
                filename = f"{timestamp.strftime('%Y%m%d%H%M')}_{p}.tif"
                filepath = os.path.join(HOURLY_MAPS_DIR, filename)
                if os.path.exists(filepath):
                    map_files[p] = os.path.basename(filepath)
            
            if map_files:
                new_metadata.append({"timestep": ts_str, "files": map_files})
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to generate map for {ts_str}. STDERR: {e.stderr}")
            continue

    with open(METADATA_PATH, 'w') as f: json.dump(new_metadata, f, indent=4)
    
    required_files = {os.path.join(HOURLY_MAPS_DIR, f) for item in new_metadata for f in item['files'].values()}
    for filename in os.listdir(HOURLY_MAPS_DIR):
        filepath = os.path.join(HOURLY_MAPS_DIR, filename)
        if filepath not in required_files and filename.endswith('.tif'):
            os.remove(filepath)
            logging.info(f"Cleaned up old hourly map: {filename}")
    
    logging.info("--- Hourly Mapping Orchestration Finished ---")

def run_daily_mapping():
    """
    Aggregates the hourly maps from the last 24 hours into a single daily map
    and uploads it to the database.
    """
    logging.info("--- Starting Daily Mapping by Aggregating Hourly Maps ---")
    today_str = date.today().isoformat()
    try:
        with open(DAILY_MAP_STATE_FILE, 'r') as f:
            if f.read().strip() == today_str:
                logging.info("Daily mapping has already run today. Skipping."); return
    except FileNotFoundError: pass

    conn = get_orchestrator_db_conn()
    if not conn: return

    try:
        end_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(hours=24)
        map_date = (end_time - timedelta(days=1)).date() # Conventionally, map for yesterday
        
        all_files = os.listdir(HOURLY_MAPS_DIR)
        pollutants_to_process = list(POLLUTANTS_MAPPING.keys()) + ['eaqi']

        for pollutant in pollutants_to_process:
            logging.info(f"Aggregating daily map for {pollutant.upper()}...")
            hourly_map_arrays = []
            
            for filename in all_files:
                if not filename.endswith(f"_{pollutant}.tif"): continue
                try:
                    ts_str = filename.split('_')[0]
                    file_ts = datetime.strptime(ts_str, '%Y%m%d%H%M').replace(tzinfo=timezone.utc)
                    if start_time <= file_ts < end_time:
                        with rasterio.open(os.path.join(HOURLY_MAPS_DIR, filename)) as src:
                            array = src.read(1).astype(np.float32)
                            array[array == src.nodata] = np.nan
                            hourly_map_arrays.append(array)
                except (ValueError, IndexError): continue
            
            if not hourly_map_arrays:
                logging.warning(f"No hourly maps found for {pollutant.upper()} in the last 24 hours."); continue

            stacked_maps = np.stack(hourly_map_arrays, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                daily_map = np.nanmean(stacked_maps, axis=0)
            
            daily_map = np.nan_to_num(daily_map, nan=raster_meta['nodata'])
            
            db_pollutant_name = POLLUTANTS_MAPPING.get(pollutant, pollutant.upper())
            upload_raster_to_postgres(daily_map.astype(raster_meta['dtype']), map_date, db_pollutant_name, conn)
            logging.info(f"Uploaded daily aggregated map for {db_pollutant_name}.")

        with open(DAILY_MAP_STATE_FILE, 'w') as f: f.write(today_str)
    finally:
        return_orchestrator_db_conn(conn)
    logging.info("--- Daily Mapping Process Finished ---")

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    logging.info("==============================================")
    logging.info("= Running Air Quality Orchestrator Script    =")
    logging.info("==============================================")
    
    init_ingestion_db()
    run_data_ingestion()
    
    try:
        get_master_metadata(MASTER_RASTER_PATH)
        initialize_orchestrator_db_pool()
    except Exception as e:
        logging.error(f"Failed to initialize master raster: {e}. Cannot proceed with mapping.")
        sys.exit(1)
    
    run_hourly_mapping_orchestration()
    #run_daily_mapping()
    
    if ingestion_db_pool: ingestion_db_pool.closeall()
    if db_pool_orchestrator: db_pool_orchestrator.closeall()
    logging.info("Database connection pools closed.")
    logging.info("Orchestrator script execution complete.")

