# This script is dedicated to fetching, cleaning, and uploading sensor data.
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import psycopg2
from psycopg2 import pool
import io
import os
import logging
from datetime import datetime, timedelta, timezone

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# -- Logging Setup --
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -- Database Configuration --
DB_CONFIG = {
    "host": "localhost", "port": 5432, "dbname": "platform_db",
    "user": "postgres", "password": "postgres"
}

# -- API Configuration --
API_BASE_URL = 'https://citylab.gate-ai.eu/sofiasensors/api/aggregated/values/parameter/stations/'
API_CHART_BASE_URL = 'https://citylab.gate-ai.eu/sofiasensors/api/aggregated/chart/measurements/'
API_USERNAME = 'testUser'
API_PASSWORD = 'test-1234-user'
API_AUTH = HTTPBasicAuth(API_USERNAME, API_PASSWORD)

# -- File & Path Configuration --
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATION_INFO_PATH = r"C:\Users\Austin\Documents\DATABANK\Masters\Thesis\Code\final_method\station_info3.csv"
CONTEXT_CACHE_PATH = os.path.join(BASE_DIR, 'context_cache.parquet')
os.makedirs(os.path.dirname(CONTEXT_CACHE_PATH), exist_ok=True)

# -- Data Parameters --
POLLUTANTS_MAPPING = {
    'Ozone': 'O3', 'Nitrogen dioxide': 'NO2', 'Sulphur dioxide': 'SO2',
    'Particulate matter 10': 'PM10', 'Particulate matter 2.5': 'PM25'
}
TOTAL_PARAMS_API = list(POLLUTANTS_MAPPING.keys())
TOTAL_PARAMS_DB = list(POLLUTANTS_MAPPING.values())

# ==============================================================================
# 2. DATABASE HELPER FUNCTIONS
# ==============================================================================
db_pool = None

def initialize_db_pool():
    """Initializes the database connection pool."""
    global db_pool
    if db_pool is None:
        db_pool = psycopg2.pool.SimpleConnectionPool(1, 5, **DB_CONFIG)

def get_db_connection():
    """Gets a connection from the pool."""
    if db_pool is None: initialize_db_pool()
    return db_pool.getconn()

def return_db_connection(conn):
    """Returns a connection to the pool."""
    if db_pool and conn: db_pool.putconn(conn)

def get_latest_timestamp(conn):
    """Queries the database to find the most recent measurement timestamp."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(measured_time) FROM sensor_data;")
            latest_time = cur.fetchone()[0]
            return latest_time.replace(tzinfo=timezone.utc) if latest_time else None
    except psycopg2.Error as e:
        logging.error(f"Database error getting latest timestamp: {e}")
        return None

def fetch_data_for_cleaning(conn, end_time):
    """
    Fetches the last 7 days of data from the DB to build the initial cleaning context.
    """
    start_time = end_time - timedelta(days=7)
    logging.info(f"Fetching data for cleaning context from {start_time} to {end_time}")
    
    query = """
    SELECT measured_time, station_id, measurement, reading_value 
    FROM sensor_data 
    WHERE measured_time BETWEEN %s AND %s
    AND measurement = ANY(%s::measurement_type[])
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, (start_time, end_time, TOTAL_PARAMS_DB))
            records = cur.fetchall()
            if not records: return pd.DataFrame()
            df = pd.DataFrame(records, columns=[desc[0] for desc in cur.description])
        
        df_pivot = df.pivot_table(
            index=['measured_time', 'station_id'],
            columns='measurement',
            values='reading_value'
        ).reset_index()
        
        # Rename columns to API format for nan_erroneous
        rename_map = {v: k for k, v in POLLUTANTS_MAPPING.items()}
        df_pivot.rename(columns=rename_map, inplace=True)
        df_pivot.rename(columns={'measured_time': 'time'}, inplace=True)
        
        station_info = pd.read_csv(STATION_INFO_PATH)
        df_merged = df_pivot.merge(station_info[['station_id', 'name']], on='station_id', how='left')
        return df_merged
    except Exception as e:
        logging.error(f"Failed to fetch data for cleaning context: {e}")
        return pd.DataFrame()

# ==============================================================================
# 3. API DATA FETCHING
# ==============================================================================

def fetch_single_hour_data(timestamp, station_info):
    """Fetches data for a single hour using the efficient batch API endpoint."""
    logging.info(f"Fetching single hour data for timestamp: {timestamp}")
    dt_str = timestamp.strftime('%Y-%m-%d %H:00')
    stations_query = ','.join(map(str, station_info['station_id'].tolist()))
    all_station_data = []

    for param_api in TOTAL_PARAMS_API:
        url = f"{API_BASE_URL}?parameter_name={requests.utils.quote(param_api)}&date={requests.utils.quote(dt_str)}&selected_stations={stations_query}"
        try:
            response = requests.get(url, auth=API_AUTH, timeout=45)
            response.raise_for_status()
            for station_id, value in response.json().items():
                if value is not None:
                    all_station_data.append({'time': timestamp, 'station_id': int(station_id), param_api: value})
        except requests.exceptions.RequestException as e:
            logging.warning(f"API request failed for single hour fetch ({param_api}): {e}")

    if not all_station_data: return pd.DataFrame()
    df = pd.DataFrame(all_station_data)
    df_wide = df.groupby(['time', 'station_id']).first().reset_index()
    return df_wide.merge(station_info[['station_id', 'name']], on='station_id', how='left')

def fetch_api_data_for_range(start_dt, end_dt, station_info):
    """Fetches data for a time range to fill gaps (e.g., after downtime)."""
    logging.info(f"Fetching API data for gap from {start_dt} to {end_dt}")
    all_data = []
    
    for _, station in station_info.iterrows():
        for param in TOTAL_PARAMS_API:
            url = f"{API_CHART_BASE_URL}?station_name={station['name']}&parameter_name={requests.utils.quote(param)}&start_date={start_dt.strftime('%Y-%m-%d')}%2000%3A00%3A00&end_date={end_dt.strftime('%Y-%m-%d')}%2023%3A59%3A59"
            try:
                response = requests.get(url, auth=API_AUTH, timeout=30)
                response.raise_for_status()
                for entry in response.json():
                    timestamp_str, value = list(entry.items())[0]
                    all_data.append({'time': pd.to_datetime(timestamp_str, utc=True), 'station_id': station['station_id'], 'name': station['name'], 'parameter': param, 'value': value})
            except requests.exceptions.RequestException as e:
                logging.warning(f"Request failed for {station['name']}, {param}: {e}")

    if not all_data: return pd.DataFrame()
    df_long = pd.DataFrame(all_data)
    df_wide = df_long.pivot_table(index=['time', 'station_id', 'name'], columns='parameter', values='value').reset_index()
    return df_wide

# ==============================================================================
# 4. DATA PROCESSING & UPLOADING
# ==============================================================================

def nan_erroneous(raw_data):
    """Cleans data by removing negative values and statistical outliers."""
    if raw_data.empty: return raw_data
    logging.info("Cleaning data for erroneous values and outliers.")
    cleaned_data = raw_data.copy()
    cleaned_data['time'] = pd.to_datetime(cleaned_data['time'], utc=True)
    cleaned_data = cleaned_data.sort_values(by=['station_id', 'time'])
    
    for col in TOTAL_PARAMS_API:
        if col in cleaned_data.columns:
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
            cleaned_data.loc[cleaned_data[col] <= 0, col] = np.nan
            
            mean = cleaned_data.set_index('time').groupby('station_id')[col].transform(lambda x: x.rolling('7D', min_periods=1).mean()).values
            std = cleaned_data.set_index('time').groupby('station_id')[col].transform(lambda x: x.rolling('7D', min_periods=1).std()).values
            
            upper_bound = mean + (3 * std)
            lower_bound = mean - (3 * std)
            
            outlier_mask = (cleaned_data[col] > upper_bound) | (cleaned_data[col] < lower_bound)
            cleaned_data.loc[outlier_mask, col] = np.nan
    return cleaned_data

def prepare_and_upload_df(df, conn):
    """Converts wide-format DataFrame to long format and uploads to the database."""
    if df.empty:
        logging.info("No new data to upload.")
        return

    logging.info(f"Preparing {len(df)} new records for database upload.")
    upload_df = df.copy()
    upload_df.rename(columns=POLLUTANTS_MAPPING, inplace=True)
    
    id_cols = ['measured_time', 'station_id']
    value_cols = [v for v in TOTAL_PARAMS_DB if v in upload_df.columns]
    
    df_long = upload_df[id_cols + value_cols].melt(id_vars=id_cols, value_vars=value_cols, var_name='measurement', value_name='reading_value')
    df_long.dropna(subset=['reading_value'], inplace=True)

    if df_long.empty:
        logging.info("Data became empty after removing NaNs. Nothing to upload.")
        return

    buffer = io.StringIO()
    df_long.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    
    try:
        with conn.cursor() as cur:
            cur.copy_expert(sql="COPY sensor_data (measured_time, station_id, measurement, reading_value) FROM STDIN WITH CSV", file=buffer)
            conn.commit()
            logging.info(f"Data successfully uploaded to the database.")
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Database upload error: {error}")
        conn.rollback()

# ==============================================================================
# 5. MAIN INGESTION ORCHESTRATION
# ==============================================================================

def run_data_ingestion():
    """Main function to orchestrate the entire data ingestion process."""
    logging.info("--- Starting Data Ingestion Process ---")
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return
    
        station_info = pd.read_csv(STATION_INFO_PATH)
        latest_db_time = get_latest_timestamp(conn)
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        if latest_db_time is None:
            latest_db_time = now - timedelta(days=7) # First run, get 7 days of context

        time_gap = now - latest_db_time
        logging.info(f"Time since last measurement: {time_gap}. Current UTC time: {now}")

        new_data_df = pd.DataFrame()
        if time_gap > timedelta(hours=1):
            logging.info("Time gap detected. Using range fetcher.")
            fetch_start = latest_db_time + timedelta(hours=1)
            new_data_df = fetch_api_data_for_range(fetch_start, now, station_info)
        elif time_gap == timedelta(hours=1):
            logging.info("Performing normal hourly fetch.")
            new_data_df = fetch_single_hour_data(now - timedelta(hours=1), station_info)
        else:
            logging.info("Data is already up-to-date. No ingestion needed.")

        if not new_data_df.empty:
            # Load historical context for better cleaning
            try:
                logging.info(f"Attempting to load context from cache: {CONTEXT_CACHE_PATH}")
                context_df = pd.read_parquet(CONTEXT_CACHE_PATH)
                context_df['time'] = pd.to_datetime(context_df['time'], utc=True)
            except FileNotFoundError:
                logging.warning("Cache not found. Fetching initial context from database.")
                context_df = fetch_data_for_cleaning(conn, latest_db_time)
                if not context_df.empty:
                    context_df.to_parquet(CONTEXT_CACHE_PATH, index=False)
                    logging.info("Initial context saved to cache.")
            
            # Combine historical context with new data before cleaning
            combined_df = pd.concat([context_df, new_data_df], ignore_index=True)
            cleaned_df = nan_erroneous(combined_df)

            # Isolate the newly cleaned data for upload
            upload_df = cleaned_df[cleaned_df['time'] > latest_db_time].copy()
            
            # Prepare a version of the uploaded data to update the cache
            upload_df_for_cache = upload_df.copy()
            
            # Rename columns for database upload
            upload_df.rename(columns={'time': 'measured_time', 'station_id': 'station_id'}, inplace=True)
            prepare_and_upload_df(upload_df, conn)

            # Update and save the context cache
            if not upload_df_for_cache.empty:
                updated_context = pd.concat([context_df, upload_df_for_cache], ignore_index=True)
                updated_context.drop_duplicates(subset=['time', 'station_id'], keep='last', inplace=True)
                
                # Trim cache to the last 7 days to keep it from growing indefinitely
                cutoff_date = now - timedelta(days=7)
                updated_context = updated_context[updated_context['time'] >= cutoff_date]
                
                updated_context.to_parquet(CONTEXT_CACHE_PATH, index=False)
                logging.info("Successfully updated context cache.")
            
    finally:
        return_db_connection(conn)
    logging.info("--- Data Ingestion Process Finished ---")

if __name__ == "__main__":
    initialize_db_pool()
    run_data_ingestion()
    if db_pool:
        db_pool.closeall()