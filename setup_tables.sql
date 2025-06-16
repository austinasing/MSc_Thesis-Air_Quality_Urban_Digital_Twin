-- Add necessesary extensions
CREATE EXTENSION IF NOT EXISTS postgis CASCADE;
CREATE EXTENSION IF NOT EXISTS postgis_raster CASCADE;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create historical sensor data table
CREATE TABLE IF NOT EXISTS sensor_data (
    id BIGSERIAL PRIMARY KEY,
    sensor_id INT NOT NULL,
    measured_time TIMESTAMPTZ NOT NULL,
    measurement TEXT NOT NULL,
    reading_value FLOAT NOT NULL,
    CONSTRAINT unique_sensor_measurement_time UNIQUE (sensor_id, measured_time, measurement)
);
-- Create timescaledb hypertable
SELECT create_hypertable('sensor_data', 'measured_time', if_not_exists => TRUE);

-- Create historical maps table
CREATE TABLE IF NOT EXISTS pollution_maps (
    id BIGSERIAL PRIMARY KEY,
    modeled_time TIMESTAMPTZ NOT NULL,
    measurement TEXT NOT NULL
    raster RASTER NOT NULL;
)
-- Create hypertable
SELECT create_hypertable('pollution_maps', 'modeled_time', if_not_exists => TRUE);
-- Add index for querying by measurement type
CREATE INDEX IF NOT EXISTS idx_pollution_maps_measurement ON pollution_maps (measurement, modeled_time DESC);