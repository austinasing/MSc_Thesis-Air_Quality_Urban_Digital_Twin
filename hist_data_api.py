# --- hist_data_api.py (Updated with fixes and debugging) ---
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import Query

app = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/platform_db")

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    """Establishes a connection to the database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to the database: {e}")
        return None

@app.get("/api/latest-station-data")
async def get_latest_pollution_data():
    """
    Fetches the most recent hour of pollution data from the database for all stations.
    """
    print("Received request for latest pollution data.")
    query = """
        WITH latest_hour_block AS (
            SELECT date_trunc('hour', MAX(measured_time)) as hour_start
            FROM sensor_data
        )
        SELECT
            sd.measured_time,
            sd.station_id,
            sd.measurement,
            sd.reading_value
        FROM
            sensor_data sd, latest_hour_block
        WHERE
            sd.measured_time >= latest_hour_block.hour_start AND
            sd.measured_time < latest_hour_block.hour_start + INTERVAL '1 hour'
            AND sd.measurement IN ('O3', 'NO2', 'SO2', 'PM10', 'PM25');
    """
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection could not be established.")
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            result = [dict(zip(colnames, row)) for row in rows]
            # ADDED: Debugging for array length
            print(f"DEBUG: Found {len(result)} records for latest-station-data.")
            return result
    except Exception as e:
        print(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        if conn:
            conn.close()

@app.get("/api/station/{station_id}")
async def get_station_chart_data(station_id: int):
    """
    Fetches historical and aggregated data for a specific station, relative to the
    latest data point in the database.
    """
    print(f"Received request for station chart data for station_id: {station_id}")
    query = """
    WITH max_time_cte AS (
        SELECT MAX(measured_time) as max_time FROM sensor_data
    ),
    last_24_hours AS (
        SELECT
            measured_time,
            measurement,
            reading_value
        FROM sensor_data, max_time_cte
        WHERE station_id = %(station_id)s
          AND measured_time >= max_time_cte.max_time - INTERVAL '24 hours'
    ),
    monthly_avg AS (
        SELECT
            EXTRACT(hour FROM measured_time)::integer as hour_of_day,
            measurement,
            AVG(reading_value) as avg_reading
        FROM sensor_data, max_time_cte
        WHERE station_id = %(station_id)s
          AND measured_time >= max_time_cte.max_time - INTERVAL '1 month'
        GROUP BY hour_of_day, measurement
    ),
    latest_readings_cte AS (
        SELECT
            measured_time, -- ADDED this field
            measurement,
            reading_value,
            ROW_NUMBER() OVER(PARTITION BY measurement ORDER BY measured_time DESC) as rn
        FROM sensor_data, max_time_cte
        WHERE station_id = %(station_id)s
          AND measured_time > max_time_cte.max_time - INTERVAL '3 hours'
    )
    SELECT
        -- UPDATED to include measured_time in the final JSON
        (SELECT json_agg(t) FROM (SELECT measured_time, measurement, reading_value FROM latest_readings_cte WHERE rn = 1 ORDER BY measurement) t) as latest,
        (SELECT json_agg(t) FROM (SELECT * FROM last_24_hours ORDER BY measured_time) t) as last_24h,
        (SELECT json_agg(t) FROM (SELECT * FROM monthly_avg ORDER BY hour_of_day) t) as monthly_avg_by_hour;
    """
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection could not be established.")
    try:
        with conn.cursor() as cur:
            cur.execute(query, {'station_id': station_id})
            result = cur.fetchone()
            if result:
                response = {
                    "latest": result[0] or [],
                    "last_24h": result[1] or [],
                    "monthly_avg_by_hour": result[2] or []
                }
                # ADDED: Debugging for array length
                print(f"DEBUG: Station {station_id} - latest: {len(response['latest'])}, 24h: {len(response['last_24h'])}, avg: {len(response['monthly_avg_by_hour'])} records.")
                return response
            return {"latest": [], "last_24h": [], "monthly_avg_by_hour": []}
    except Exception as e:
        print(f"Error executing query for station {station_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        if conn:
            conn.close()


@app.get("/api/historical/building")
async def get_historical_building_data(
    lat: float = Query(..., description="Latitude of the building"),
    lon: float = Query(..., description="Longitude of the building")
):
    """Fetches historical raster data for a specific point (building)."""
    # CHANGED: Replaced NOW() with a CTE for the max modeled date for robustness.
    query = """
    WITH max_date_cte AS (
        SELECT MAX(modeled_date) as max_date FROM daily_historical_maps
    ),
    point_geom AS (
        SELECT ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326) as geom
    ),
    daily_data AS (
        SELECT
            dhm.modeled_date::text,
            dhm.measurement,
            ST_Value(dhm.raster_data, pg.geom) as reading_value
        FROM daily_historical_maps dhm, point_geom pg, max_date_cte
        WHERE
            dhm.modeled_date >= max_date_cte.max_date - INTERVAL '14 days'
            AND ST_Intersects(dhm.raster_data, pg.geom)
    ),
    weekly_data AS (
        SELECT
            time_bucket('1 week', dhm.modeled_date)::date::text as week_start,
            dhm.measurement,
            AVG(ST_Value(dhm.raster_data, pg.geom)) as avg_reading_value
        FROM daily_historical_maps dhm, point_geom pg, max_date_cte
        WHERE
            dhm.modeled_date >= max_date_cte.max_date - INTERVAL '1 year'
            AND ST_Intersects(dhm.raster_data, pg.geom)
        GROUP BY week_start, dhm.measurement
    )
    SELECT
        (SELECT json_agg(t) FROM (SELECT * FROM daily_data) t) as last_two_weeks,
        (SELECT json_agg(t) FROM (SELECT * FROM weekly_data) t) as last_year_weekly;
    """
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection could not be established.")
    try:
        with conn.cursor() as cur:
            cur.execute(query, {'lat': lat, 'lon': lon})
            result = cur.fetchone()
            if result:
                response = {"daily_last_two_weeks": result[0] or [], "weekly_last_year": result[1] or []}
                 # ADDED: Debugging for array length
                print(f"DEBUG: Building at {lat},{lon} - daily: {len(response['daily_last_two_weeks'])}, weekly: {len(response['weekly_last_year'])} records.")
                return response
            return {"daily_last_two_weeks": [], "weekly_last_year": []}
    except Exception as e:
        print(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        if conn:
            conn.close()


@app.get("/api/historical/lez")
async def get_lez_data():
    """Fetches historical data for the Low Emission Zone for specific winter periods."""
    query = """
    SELECT
        week::date::text as week_start,
        region_name,
        measurement,
        avg_value
    FROM weekly_region_averages
    WHERE
        region_name = 'sofia_lez' AND (
            (week >= '2022-12-01' AND week < '2023-03-01') OR
            (week >= '2023-12-01' AND week < '2024-03-01') OR
            (week >= '2024-12-01' AND week < '2025-03-01')
        )
    ORDER BY week_start, measurement;
    """
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection could not be established.")
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            result = [dict(zip(colnames, row)) for row in rows]
            # ADDED: Debugging for array length
            print(f"DEBUG: Found {len(result)} records for LEZ data.")
            return result
    except Exception as e:
        print(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        if conn:
            conn.close()

@app.get("/api/historical/district")
async def get_district_data():
    """Fetches historical and aggregated data for all districts."""
    # CHANGED: Replaced NOW() with a CTE for the max modeled date for robustness.
    query = """
    WITH max_date_cte AS (
    -- Using the daily aggregate is faster than hitting the raw hypertable.
    SELECT MAX(day) as max_date FROM daily_region_averages
)
SELECT
    (
        SELECT json_agg(t)
        FROM (
            SELECT
                day::date::text as modeled_date,
                region_name,
                measurement,
                avg_value
            FROM daily_region_averages, max_date_cte
            WHERE
                region_type = 'district'
                AND day >= max_date_cte.max_date - INTERVAL '14 days'
            ORDER BY day
        ) t
    ) as daily_by_district,
    (
        SELECT json_agg(t)
        FROM (
            SELECT
                day::date::text as modeled_date,
                measurement,
                AVG(avg_value) as overall_avg_value
            FROM daily_region_averages, max_date_cte
            WHERE
                region_type = 'district'
                AND day >= max_date_cte.max_date - INTERVAL '14 days'
            GROUP BY day, measurement
            ORDER BY day
        ) t
    ) as daily_overall_average,
    (
        SELECT json_agg(t)
        FROM (
            SELECT
                week::date::text as week_start,
                region_name,
                measurement,
                avg_value
            FROM weekly_region_averages, max_date_cte
            WHERE
                region_type = 'district'
                AND week >= time_bucket('1 week', max_date_cte.max_date - INTERVAL '1 year')
            ORDER BY week_start
        ) t
    ) as weekly_by_district,
    (
        SELECT json_agg(t)
        FROM (
            SELECT
                week::date::text as week_start,
                measurement,
                AVG(avg_value) as overall_avg_value
            FROM weekly_region_averages, max_date_cte
            WHERE
                region_type = 'district'
                AND week >= time_bucket('1 week', max_date_cte.max_date - INTERVAL '1 year')
            GROUP BY week, measurement
            ORDER BY week_start
        ) t
    ) as weekly_overall_average;
    """
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection could not be established.")
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()
            if result:
                response = {
                    "daily_by_district": result[0] or [],
                    "daily_overall_average": result[1] or [],
                    "weekly_by_district": result[2] or [],
                    "weekly_overall_average": result[3] or []
                }
                # ADDED: Debugging for array length
                print(f"DEBUG: Districts - daily: {len(response['daily_by_district'])}, weekly: {len(response['weekly_by_district'])} records.")
                return response
            return {
                "daily_by_district": [], "daily_overall_average": [],
                "weekly_by_district": [], "weekly_overall_average": []
            }
    except Exception as e:
        print(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        if conn:
            conn.close()

@app.get("/")
def read_root():
    return {"status": "API is running"}