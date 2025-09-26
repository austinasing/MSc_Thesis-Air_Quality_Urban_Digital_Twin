-- Drop old tables and types if they exist
DROP TABLE IF EXISTS sensor_data;
DROP TABLE IF EXISTS pollution_maps;
DROP TABLE IF EXISTS precalc_ivs;
DROP TYPE IF EXISTS measurement_type;
DROP TYPE IF EXISTS map_type;
DROP TYPE IF EXISTS buffer_type;

-- Add necessesary extensions
CREATE EXTENSION IF NOT EXISTS postgis CASCADE;
CREATE EXTENSION IF NOT EXISTS postgis_raster CASCADE;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Potentially create lookup table for stations here if I want to work with more metadata

-- Create historical sensor data table
CREATE TYPE measurement_type as ENUM ('O3','NO2','SO2','PM10','PM25','T','P','RH','WD','WS');
CREATE TABLE sensor_data (
    measured_time TIMESTAMPTZ NOT NULL,
    station_id SMALLINT NOT NULL,
    measurement measurement_type NOT NULL,
    reading_value REAL NOT NULL,
    id SERIAL,
    PRIMARY KEY (measured_time, station_id, measurement)
);
-- Create timescaledb hypertable
SELECT create_hypertable('sensor_data', 'measured_time', if_not_exists => TRUE);

-- Create precalculated IV table
CREATE TYPE buffer_type as ENUM('cir','NE','N','NW','W','SW','S','SE','E', 'Zero');
CREATE TABLE precalc_ivs (
    indep_var TEXT NOT NULL,
    buffer_type buffer_type,
    radius SMALLINT,
    raster_data RASTER NOT NULL,
    id SERIAL PRIMARY KEY
);

-- Create hourly historical table
CREATE TYPE map_type as ENUM ('O3', 'NO2', 'SO2', 'PM10', 'PM25', 'EAQI', 'DOM');
-- Create historical maps table --
CREATE TABLE daily_historical_maps (
    modeled_date DATE NOT NULL,
    measurement map_type NOT NULL,
    raster_data RASTER NOT NULL,
    id SERIAL,
    PRIMARY KEY (modeled_date, measurement)
);
-- Create hypertable
SELECT create_hypertable('daily_historical_maps', 'modeled_date', if_not_exists => TRUE);
-- Add index for quick recent querying by measurement type
CREATE INDEX IF NOT EXISTS idx_daily_measurement ON daily_historical_maps (modeled_date DESC);
-- Spatial index for ad-hoc spatial queries (define own polygon)
CREATE INDEX IF NOT EXISTS idx_daily_rast_gist ON daily_historical_maps USING GIST (ST_ConvexHull(raster_data));

-- Create table with analysis regions (GeoJSON)
CREATE TABLE analysis_regions (
    region_id SERIAL PRIMARY KEY,
    region_name TEXT NOT NULL UNIQUE,
    region_type TEXT,
    geom GEOMETRY(MultiPolygon, 4326) NOT NULL
);
-- Add spatial index
CREATE INDEX IF NOT EXISTS idx_analysis_regions_geom_gist ON analysis_regions USING GIST (geom);

-- Create the continuous aggregate for weekly regional averages
CREATE MATERIALIZED VIEW weekly_region_averages
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 week', d.modeled_date) AS week,
    r.region_id,
    r.region_name,
    r.region_type, -- Include the new type column
    d.measurement,
    AVG((ST_SummaryStats(ST_Clip(d.raster_data, r.geom))).mean) AS avg_value
FROM
    daily_historical_maps AS d,
    analysis_regions AS r
WHERE
    ST_Intersects(d.raster_data, r.geom)
GROUP BY
    week, r.region_id, r.region_name, r.region_type, d.measurement;

-- Add a policy to keep it automatically updated
SELECT add_continuous_aggregate_policy('weekly_region_averages',
    start_offset => INTERVAL '1 month',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- Step 1: Create a Daily Continuous Aggregate
-- This new materialized view will pre-calculate the daily average for each region.
CREATE MATERIALIZED VIEW daily_region_averages
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', d.modeled_date) AS day,
    r.region_id,
    r.region_name,
    r.region_type,
    d.measurement,
    AVG((ST_SummaryStats(ST_Clip(d.raster_data, r.geom))).mean) AS avg_value
FROM
    daily_historical_maps AS d,
    analysis_regions AS r
WHERE
    ST_Intersects(d.raster_data, r.geom)
GROUP BY
    day, r.region_id, r.region_name, r.region_type, d.measurement;

-- Add a policy to keep the new daily aggregate updated automatically.
SELECT add_continuous_aggregate_policy('daily_region_averages',
    start_offset => INTERVAL '1 month',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- Create a standard materialized view for weekly averaged rasters
CREATE MATERIALIZED VIEW weekly_historical_maps AS
SELECT
    time_bucket('1 week', modeled_date) AS week,
    measurement,
    ST_Union(raster_data, 'MEAN') AS raster_data
FROM
    daily_historical_maps
GROUP BY
    week, measurement;

-- Add indexes for fast lookups
CREATE UNIQUE INDEX ON weekly_historical_maps (week);
CREATE INDEX ON weekly_historical_maps USING GIST (ST_ConvexHull(raster_data));