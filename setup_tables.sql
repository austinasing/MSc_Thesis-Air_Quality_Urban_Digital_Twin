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
CREATE TYPE buffer_type as ENUM('C','NE','N','NW','W','SW','S','SE','E');
CREATE TABLE precalc_ivs (
    indep_var TEXT NOT NULL,
    buffer_type buffer_type,
    radius SMALLINT,
    raster_data RASTER NOT NULL,
    id SERIAL PRIMARY KEY
);

-- Create historical maps table
CREATE TYPE map_type as ENUM ('AQI', 'O3', 'NO2', 'SO2', 'PM10', 'PM25');
CREATE TABLE pollution_maps (
    modeled_time TIMESTAMPTZ NOT NULL,
    measurement map_type NOT NULL,
    raster_data RASTER NOT NULL,
    id SERIAL,
    PRIMARY KEY (modeled_time, measurement)
);
-- Create hypertable
SELECT create_hypertable('pollution_maps', 'modeled_time', if_not_exists => TRUE);
-- Add index for quick recent querying by measurement type
CREATE INDEX IF NOT EXISTS idx_pollution_maps_measurement ON pollution_maps (measurement, modeled_time DESC);

-- For future:
-- Set up table for predefined analysis areas
-- Continuous aggregates to already be doing the work