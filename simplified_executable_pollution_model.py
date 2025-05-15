# Install module dependencies
import numpy as np
import pandas as pd
import os
import requests
from requests.auth import HTTPBasicAuth
import certifi
import pyproj
from pyproj import Transformer
from shapely.geometry import Point
from shapely.affinity import translate
from shapely.ops import nearest_points
from shapely.strtree import STRtree
import rasterio
import rasterio.transform
from rasterio.transform import from_bounds 
from rasterio import features
from rasterio.crs import CRS
import geopandas as gpd
from functools import partial
import concurrent.futures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import warnings
warnings.simplefilter("ignore", ConvergenceWarning)
# Set-up logging handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class AirPollutionModel:
    def __init__(self, pollutant, resolution):
        '''Initialize air pollution model for regression mapping'''
        # Model configuration
        self.pollutant = pollutant
        self.resolution = resolution
        self.API_credentials = # not public
        # Spatial data files (initialized later)
        self.building_height_file = None
        self.bus_stops_file = None
        self.elevation_file = None
        self.landuse_file = None
        self.major_road_raster_file = None
        self.minor_road_file = None
        self.major_road_vector_file = None
        self.ndvi_file = None
        self.population_file = None
        self.boundary_file = None
        # Model data (initialized during processing)
        self.stations = None
        self.indep_var_info = None
        self.indep_var_calculations = None
        self.model = None
        self.scaler = None
        self.coefficients = None
        # Prediction map data
        self.rasterized_boundary = None
        self.prediction_map_profile = None
        self.non_nan_indices = None
        self.boundary_points = None
        # Initalize data and model components
        self._initialize_spatial_data()
        self._initialize_prediction_map()
        self._initialize_pollution_stations()

    def _initialize_spatial_data(self):
        '''Set up the file paths for the downloaded data'''
        self.data_folder = Path('spatial_data')
        self.building_height_file = self.data_folder / 'build_height.tif'
        self.bus_stops_file = self.data_folder / 'bus_stops.gpkg'
        self.elevation_file = self.data_folder / 'elevation.tif'
        self.landuse_file = self.data_folder / 'landuse.tif'
        self.major_road_raster_file = self.data_folder / 'major_rd.tif'
        self.minor_road_file = self.data_folder / 'minor_rd.tif'
        self.major_road_vector_file = self.data_folder / 'major_rd.gpkg'
        self.ndvi_file = self.data_folder / 'ndvi.tif'
        self.population_file = self.data_folder / 'pop.tif'
        self.boundary_file = self.data_folder / 'boundary.gpkg'

        # Name the independant variables to be calculated
        self.indep_var_info =[{'var':'lu_urban', 'var_data':self.landuse_file, 'buffer':1},
              {'var':'lu_grass', 'var_data':self.landuse_file, 'buffer':1},
              {'var':'lu_forest', 'var_data':self.landuse_file, 'buffer':1},
              {'var':'build_cover', 'var_data':self.building_height_file, 'buffer':1},
              {'var':'build_vol', 'var_data':self.building_height_file, 'buffer':1},
              {'var':'build_std', 'var_data':self.building_height_file, 'buffer':1},
              {'var':'ndvi', 'var_data':self.ndvi_file, 'buffer':1},
              {'var':'elevation', 'var_data':self.elevation_file, 'buffer':1},
              {'var':'pop', 'var_data':self.population_file, 'buffer':1},
              {'var':'mj_road_ln', 'var_data':self.major_road_raster_file, 'buffer':1},
              {'var':'mi_road_ln', 'var_data':self.minor_road_file, 'buffer':1},
              {'var':'bus_stops', 'var_data':self.bus_stops_file, 'buffer':1},
              {'var':'mj_road_dis', 'var_data':self.major_road_vector_file, 'buffer':0}
           ]
        
        # Set projection details (due to PROJ errors that may be specific to my computer)
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
        self.project_crs = CRS.from_wkt(wkt_7801)
        os.environ["GTIFF_SRS_SOURCE"] = "EPSG"
        os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()

    def _initialize_prediction_map(self):
        '''Creates the prediction map points based on the boundary file and chosen map resolution'''
        gdf = gpd.read_file(self.boundary_file)
        # Give boundary geometry the value 1 to burn in the raster
        shapes = [(geom, 1) for geom in gdf.geometry]
        # Set raster details
        minx, miny, maxx, maxy = gdf.total_bounds
        width = int((maxx - minx) / self.resolution)
        height = int((maxy - miny) / self.resolution)
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        # Initialize empty raster
        fill_value = 0
        raster = np.full((height,width), fill_value, dtype=np.float32)
        # Rasterize the boundary polygon
        self.rasterized_boundary = features.rasterize(
            shapes=shapes,
            out=raster,
            transform=transform,
            fill=fill_value,
            all_touched=False,
            dtype=np.float32
        )
        # Create profile of the raster for writing the prediction map
        self.prediction_map_profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'float32',
            'crs': self.project_crs,
            'transform': transform,
            'nodata': -9999
        }
        # Identify valid cells in the raster to only map values within the boundary
        self.non_nan_indices = np.where(self.rasterized_boundary != fill_value)
        rows, cols = self.non_nan_indices
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        self.boundary_points = [Point(x, y) for x, y in zip(xs, ys)]
        logging.info(f'{self.resolution}m prediction map initialized. Dimensions: {width} x {height}. Total points within boundary: {len(self.boundary_points)}')
        
    def _initialize_pollution_stations(self):
        '''Collects the stations and their locations to be used in the model'''
        # Send request to the GATE API
        try:
            station_url = 'https://citylab.gate-ai.eu/sofiasensors/api/stations/'
            response = requests.get(station_url, auth=HTTPBasicAuth(*self.API_credentials), verify=certifi.where())
            response.raise_for_status()
            stations_data = response.json()
            # Save the important information
            all_stations_df = pd.DataFrame(stations_data, columns=['name', 'longitude', 'latitude', 'operator'])          
            # Convert Lat/Lon values into EPSG7801
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:7801", always_xy=True)
            def reproject_coords(row):
                '''Reprojects coordinates in a dataframe row'''
                longitude, latitude = row['longitude'], row['latitude']
                x, y = transformer.transform(longitude, latitude)
                x, y = round(x, ndigits=1), round(y, ndigits=1) # Round the coordinates to decimeters
                return (x,y)
            all_stations_df['location'] = all_stations_df.apply(reproject_coords, axis=1)
            all_stations_df.drop(['latitude', 'longitude'], axis=1, inplace=True)
            # Remove ExEA stations (don't have all pollutants)
            self.stations = all_stations_df[all_stations_df['operator'] != 'Executive environmental agency (ExEA)']
            logging.info(f'Pollution station info gathered. {len(self.stations)} total stations')
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to initialize pollution stations: {e}")

    # Core spatial calculation functions used in independant variable calculation and prediction mapping
    def _create_mask(self, raster_info, buffer_size):
        '''Creates a circular mask of the buffer size according the raster details'''
        pixel_size = raster_info['resolution']
        radius_px = int(np.round(buffer_size / pixel_size))
        size = 2 * radius_px + 1 # Size of square to hold the circle
        center = radius_px # Center of the square is at radius,radius
        y,x = np.ogrid[:size, :size]
        distance_squared = (x - center) ** 2 + (y - center) ** 2 # Calculate grid of distance from the center
        return distance_squared <= radius_px ** 2 # Mask the distance grid for only values within the radius
    def _get_cell_indices(self, point, raster_info): 
        '''Returns the index of a raster for a particular point'''
        xmin = raster_info['xmin']
        ymax = raster_info['ymax']
        resolution = raster_info['resolution']
        pointx, pointy = point.x, point.y
        j = ((pointx - xmin) / resolution)
        i = ((ymax - pointy) / resolution)
        return round(i), round(j)
    def _get_buffer_cells(self, point_index, array, mask):
        '''Collects cells from an array for a point and mask'''
        rows, cols = array.shape
        point_r, point_c = point_index
        mask_size = mask.shape[0]
        radius = mask_size // 2
        # Calculate bounds for the slice in the main array
        row_start = max(0, point_r - radius)
        row_end = min(rows, point_r + radius + 1)
        col_start = max(0, point_c - radius)
        col_end = min(cols, point_c + radius + 1)
        # Bounds for the mask slice to align with array slice
        mask_row_start = max(0, radius - point_r)
        mask_row_end = mask_size - max(0, (point_r + radius + 1) - rows)
        mask_col_start = max(0, radius - point_c)
        mask_col_end = mask_size - max(0, (point_c + radius + 1) - cols)
        # Extract the slices
        array_slice = array[row_start:row_end, col_start:col_end]
        mask_slice = mask[mask_row_start:mask_row_end, mask_col_start:mask_col_end]
        return array_slice[mask_slice]
    def _calculate_raster_buffer(self, point, var, raster_info, mask):
        '''Calculate raster-based buffer variable for single point'''
        point_index = self._get_cell_indices(point, raster_info)
        buffer_cells = self._get_buffer_cells(point_index, raster_info['array'], mask)
        total_cells = buffer_cells.size
        nodata_val = raster_info['nodata']
        valid_cells = buffer_cells[buffer_cells != nodata_val]
        # Calculation depending on variable
        if var == 'lu_urban':
            value = np.isin(valid_cells, 1).sum()
        elif var == 'lu_grass':
            value = np.isin(valid_cells, 2).sum()
        elif var == 'lu_forest':
            value = np.isin(valid_cells, 3).sum()
        elif var in ('build_cover', 'build_vol', 'build_std'):
            count = np.count_nonzero(valid_cells)
            # Building cover = % buildings over buffer size
            if var == 'build_cover':
                if total_cells != 0: # Cover edge case where total cells = 0
                    value = count / total_cells
                else:
                    value = 0
            # Building volume = building height * area
            elif var == 'build_vol':
                value = np.sum(valid_cells) * count * 10
            # Building height variation = standard deviation of building heights
            elif var == 'build_std':
                if len(valid_cells) >= 2: # Can't have deviation between less than 2 data points
                    value = np.std(valid_cells)
                else:
                    value = 0
        # Elevation, population, and ndvi are all average values in the buffer
        elif var in ['elevation', 'pop', 'ndvi']:
            value = float(np.mean(valid_cells))
        # Roads and parking lots: count total valid cells
        elif var in ('mj_road_ln', 'mi_road_ln'):
            value = np.count_nonzero(valid_cells)
        return value
    def _calculate_vector_buffer(self, point, vector_data, buffer_template):
        '''Calculate vector-based buffer value for a single point'''
        point_buffer = translate(buffer_template, xoff=point.x, yoff=point.y)
        inner_points = vector_data[vector_data.geometry.within(point_buffer)]
        return len(inner_points)
    def _distance_calculations(self, point, tree, geoms):
        '''Calculate the distance between the point being calculated and the independant variable'''
        nearest_index = tree.nearest(point)
        nearest_geom = geoms[nearest_index]
        p1, p2 = nearest_points(point, nearest_geom)
        return p1.distance(p2)
    def _parallel_compute(self, func, input_points):
        '''Parallel compute a function with multiple points'''
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(func, input_points))
        return results
    def _calculate_points(self, points, var_info, buffer_size):
        var = var_info['var']
        data_file = var_info['var_data']
        if data_file.suffix == '.tif':
            with rasterio.open(data_file) as src:
                raster_info = {'array':src.read(1), 'resolution':src.res[0], 'xmin':src.bounds.left, 'ymax':src.bounds.top, 'nodata':src.nodata}
                mask = self._create_mask(raster_info, buffer_size) 
                # Define the partial function pre-filled with arguments
                func = partial(self._calculate_raster_buffer, var=var, raster_info=raster_info, mask=mask)
                values = self._parallel_compute(func, points)
        if data_file.suffix == '.gpkg':
            vector = gpd.read_file(data_file)
            if var == 'bus_stops':
                # Create buffer template geometry to use for each point
                buffer_template = Point(0,0).buffer(buffer_size, quad_segs=2)
                # Define the partial function pre-filled with arguments
                func = partial(self._calculate_vector_buffer, vector_data=vector, buffer_template=buffer_template)
                values = self._parallel_compute(func, points)
            else: # Otherwise calculate distance
                # Create STRtree spatial index from vector
                geoms = vector.geometry.values 
                tree = STRtree(geoms)
                func = partial(self._distance_calculations, tree=tree, geoms=geoms)
                values = self._parallel_compute(func, points)
        return values

    def calculate_independent_vars(self):
        '''Calculate the variable values for all of the stations'''
        station_points = [Point(loc) for loc in self.stations['location']]
        self.indep_var_calculations = self.stations[['name']].copy()
        # Process each independant variable with multiple buffer sizes
        buffer_sizes = [100,500,1000,2000,3000] # meters
        for var_info in self.indep_var_info:
            var = var_info['var']
            use_buffer = var_info['buffer'] == 1
            if use_buffer:
                for buffer_size in buffer_sizes:
                    calc_name = f'{var}_{buffer_size}m'
                    values = self._calculate_points(station_points, var_info, buffer_size)
                    self.indep_var_calculations[calc_name] = values
            else:
                calc_name = var
                values = self._calculate_points(station_points, var_info, None)
                self.indep_var_calculations[calc_name] = values
        logging.info(f'Calculated {len(self.indep_var_calculations.columns.to_list()) - 1} potential independant variables')
        return self.indep_var_calculations
        
    def get_measurements(self, timestamp = None):
        '''Get sensor data from specific timestamp, gets earliest measurement if no timestamp is provided'''
        logging.info('Collecting measurements from API...')
        measurements = self.stations[['name']].copy()
        if timestamp:
            # Set start time from timestamp and end time one hour later
            s_time = datetime.strptime(timestamp, "%Y-%m-%d %H:00")
            s_date = datetime.strftime(s_time, "%Y-%m-%d")
            s_hour = datetime.strftime(s_time, "%H")
            e_time = s_time + timedelta(hours=1)
            e_date = datetime.strftime(e_time, "%Y-%m-%d")
            e_hour = datetime.strftime(e_time, "%H")
            values = []
            for _, station in tqdm(self.stations.iterrows(), total=len(self.stations)):
                station_name = station['name']
                url = f'https://citylab.gate-ai.eu/sofiasensors/api/aggregated/values/hour/station/?station_name={station_name}&selected_params={self.pollutant}&start_date={s_date}%20{s_hour}%3A00%3A00&end_date={e_date}%20{e_hour}%3A00%3A00&calculation_type=Mean'
                try:
                    response = requests.get(url, auth=HTTPBasicAuth(*self.API_credentials))
                    response.raise_for_status()
                    data = response.json()
                    if data:
                        values.append(data[0]['Values'])
                    else:
                        logging.error(f'No data for {station_name}, {self.pollutant}')
                        values.append(None)
                except requests.exceptions.RequestException as e:
                    logging.error(f"Request failed for station {station_name}: {e}")
                    values.append(None)
            measurements[self.pollutant] = values
        else:
            url = 'https://citylab.gate-ai.eu/sofiasensors/api/stations/lastmeasurements'
            try:
                response = requests.get(url, auth=HTTPBasicAuth(*self.API_credentials))
                data = response.json()
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
            values = {}
            for entry in data:
                if self.pollutant in entry['measurements'].keys():
                    # Assigns pollutant value to station name
                    values[entry['station_name']] = entry['measurements'][self.pollutant]
            # Maps the measurements to the dataframe
            measurements[self.pollutant] = measurements['name'].map(values)
            timestamp = data[0]["date_measured"]
        # Remove stations without measurements from the dataframe
        measurements = measurements[~measurements[self.pollutant].isna()]
        logging.info(f'{self.pollutant} measurements for {timestamp} GMT collected. Values from {len(values)} stations')
        return measurements

    def calculate_model_coefficients(self, measurements):
        '''Train the model using the measurements and indep var calculations to get coefficients'''
        # Merge sensor data with computed indep variables
        training_data = measurements.merge(self.indep_var_calculations, on='name', how='left')
        pollutant_buffer_cols = [col for col in self.indep_var_calculations.columns if col not in ['name']]
        X = training_data[pollutant_buffer_cols]
        y = training_data[self.pollutant]
        # Scale features (common step for regularization models like Lasso)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Save scaling info after fitting to un-scale prediction maps in mapping step
        feature_means = scaler.mean_
        feature_stds = scaler.scale_
        self.scaling_params = {}
        for i, feature in enumerate(X.columns):
            self.scaling_params[feature] = {'mean':feature_means[i],'std':feature_stds[i]}
        # Train model
        alphas = [0.5 + i*0.1 for i in range(11)] # 0.5 - 1.5 to encourage variable selection
        model = LassoCV(
            alphas=alphas, # Intensity of regularization penalty (0 = no regularization, normal OLS)
            cv=5, # Split the training data into 5 train/test segments to test
            random_state=42,
            max_iter=10000,
            tol=1e-4
        )
        model.fit(X_scaled, y)
        # Store intercept and non-zero coefficients
        coef_dict = {'intercept': model.intercept_,
                     **{coef_name: value for coef_name, value in zip(pollutant_buffer_cols, model.coef_) if abs(value) > 1e-8}}
        logging.info(f'Optimal model calculated with alpha: {round(model.alpha_, 2)}, {len(coef_dict) -1} variables')
        return coef_dict

    def generate_prediction(self, coefficients):
        '''Use model prediction to make base prediction map (stored in 1D array)'''
        # Initialize prediction array
        prediction = np.zeros(len(self.boundary_points))
        logging.info('Generating prediction map...')
        # Process each selected coefficient
        for coef_name, coef_value in tqdm(coefficients.items()):
            if coef_name == 'intercept':
                continue
            # Parse variable and buffer size from the coefficient name
            if '0m' in coef_name: # Coefficient with buffer
                var, buffer_str = coef_name.rsplit('_',1)
                buffer_size = int(buffer_str[:-1])
            else:
                var = coef_name
                buffer_size = None
            var_info = next((d for d in self.indep_var_info if d.get('var') == var))
            # Process all map points for this variable
            values = self._calculate_points(self.boundary_points, var_info, buffer_size)
            # Re-scale the values using the scaling parameters derived in training stage
            scaled_values = (values - self.scaling_params[coef_name]['mean']) / self.scaling_params[coef_name]['std']
            # Apply coefficient and add to prediction
            prediction += np.array(scaled_values) * coef_value
        prediction += coefficients['intercept']
        # Set values below 0 to be 0 (predictions aren't always realistic and this helps visualization)
        prediction[prediction < 0] = 0 
        logging.info('Prediction map calculated')
        return prediction
    
    def visualize_prediction_map(self, prediction):
        # Create 2D visualization
        height, width = self.rasterized_boundary.shape
        vis_raster = np.full((height, width), np.nan)
        # Place prediction values into 2D array
        rows, cols = self.non_nan_indices
        vis_raster[rows, cols] = prediction
        # Create figure and plot
        plt.figure(figsize=(10, 8))
        plt.imshow(vis_raster, cmap='RdYlGn_r')
        plt.colorbar(label=f'{self.pollutant} concentration')
        plt.title(f'Predicted {self.pollutant} Concentration')
        plt.axis('off')
        return plt
        
    def save_output(self, prediction, output_path):
        '''Save prediction map to GeoTIFF'''
        height, width = self.rasterized_boundary.shape
        output_raster = np.full((height, width),
                                self.prediction_map_profile['nodata'],
                                dtype=np.float32)
        # Put 1D values back into 2D raster at non-nan indices
        rows, cols = self.non_nan_indices
        output_raster[rows,cols] = prediction

        with rasterio.open(output_path, 'w', **self.prediction_map_profile) as dst:
            dst.write(output_raster, 1)
        print(f'Prediction map saved to: {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Air Pollution Regression Model')
    parser.add_argument('--pollutant', required=True, choices=['O3', 'NO2', 'SO2', 'PM10', 'PM2.5'],
                      help='Pollutant to model')
    parser.add_argument('--timestamp', required=False, type=str,
                      help='Timestamp in format YYYY-MM-DD HH:00 (default: most recent measurements)')
    parser.add_argument('--resolution', required=False, type=int, default=100,
                      help='Spatial resolution in meters for prediction map (default: 100)')
    parser.add_argument('--output', required=False, 
                      help='Output file path for prediction map (default: no file output)')
    parser.add_argument('--visualize', action='store_true',
                      help='Display visualization of the prediction map')
    args = parser.parse_args()
    
    if not args.output and not args.visualize:
        logging.warning('Neither --output nor --visualize specified. No results will be produced.')
        logging.warning('Please specify at least one of --output or --visualize to see results.')
        exit(1)
        
    # Create model and run prediction
    model = AirPollutionModel(args.pollutant, args.resolution)
    model.calculate_independent_vars()
    measurements = model.get_measurements(args.timestamp)
    coefficients = model.calculate_model_coefficients(measurements)
    prediction = model.generate_prediction(coefficients)
    
    if args.output:
        model.save_output(prediction, args.output)
    if args.visualize:
        plt = model.visualize_prediction_map(prediction)
        plt.show()