# TerriaMap Catalogue Generator & WMS Proxy API
import json
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import xml.etree.ElementTree as ET
import httpx
import matplotlib.colors as mcolors
import numpy as np
from cachetools import LRUCache

tile_cache = LRUCache(maxsize=512)

# --- Configuration ---
METADATA_FILE_PATH = r'C:\Users\Austin\Documents\DATABANK\Masters\Thesis\Code\final_method\web_maps_hourly\metadata_lookup.json'
BASE_COG_URL = "http://localhost:8001"
TITILER_ENDPOINT = "http://localhost:8000"
SELF_URL = "http://localhost:8002"

COLORS = ['#50f0e6','#50ccaa','#f0e641','#ff5050','#960032', '#7d2181']
EAQI_LIMITS = {
    'no2': [0, 40, 90, 120, 230, 340, 1000],
    'o3': [0, 50, 100, 130, 240, 380, 800],
    'pm10': [0, 20, 40, 50, 100, 150, 1200],
    'pm25': [0, 10, 20, 25, 50, 75, 800],
    'so2': [0, 100, 200, 350, 500, 750, 1250],
    'eaqi': [0,1,2,3,4,5,6]
}
POLLUTANT_LEGENDS = {
    'o3':{
        'title':"Ozone (O3) µg/m³",
        'items': [
            {'color':'#50f0e6','title':'0 - 50 (Good)'},
            {'color':'#50ccaa','title':'50 - 100 (Fair)'},
            {'color':'#f0e641','title':'100 - 130 (Moderate)'},
            {'color':'#ff5050','title':'130 - 240 (Poor)'},
            {'color':'#960032','title':'240 - 380 (Very Poor)'},
            {'color':'#7d2181','title':'380+ (Extremely Poor)'},
        ]
    },
    'no2':{
        'title':"Nitrogen Dioxide (NO2) µg/m³",
        'items': [
            {'color':'#50f0e6','title':'0 - 40 (Good)'},
            {'color':'#50ccaa','title':'40 - 90 (Fair)'},
            {'color':'#f0e641','title':'90 - 120 (Moderate)'},
            {'color':'#ff5050','title':'120 - 230 (Poor)'},
            {'color':'#960032','title':'340 - 1000 (Very Poor)'},
            {'color':'#7d2181','title':'1000+ (Extremely Poor)'},
        ]
    },
    'so2':{
        'title':"Sulphur Dioxide (SO2) µg/m³",
        'items': [
            {'color':'#50f0e6','title':'0 - 100 (Good)'},
            {'color':'#50ccaa','title':'100 - 200 (Fair)'},
            {'color':'#f0e641','title':'200 - 350 (Moderate)'},
            {'color':'#ff5050','title':'350 - 500 (Poor)'},
            {'color':'#960032','title':'500 - 750 (Very Poor)'},
            {'color':'#7d2181','title':'750+ (Extremely Poor)'},
        ]
    },
    'pm10':{
        'title':"Particulate Matter <10µm (PM10) µg/m³",
        'items': [
            {'color':'#50f0e6','title':'0 - 20 (Good)'},
            {'color':'#50ccaa','title':'20 - 40 (Fair)'},
            {'color':'#f0e641','title':'40 - 50 (Moderate)'},
            {'color':'#ff5050','title':'50 - 100 (Poor)'},
            {'color':'#960032','title':'100 - 150 (Very Poor)'},
            {'color':'#7d2181','title':'150+ (Extremely Poor)'},
        ]
    },
    'pm25':{
        'title':"Particulate Matter <2.5µm (PM25) µg/m³",
        'items': [
            {'color':'#50f0e6','title':'0 - 10 (Good)'},
            {'color':'#50ccaa','title':'10 - 20 (Fair)'},
            {'color':'#f0e641','title':'20 - 25 (Moderate)'},
            {'color':'#ff5050','title':'25 - 50 (Poor)'},
            {'color':'#960032','title':'50 - 75 (Very Poor)'},
            {'color':'#7d2181','title':'75+ (Extremely Poor)'},
        ]
    },
    'eaqi':{
        'title':"European Air Quality Index",
        'items': [
            {'color':'#50f0e6','title':'1 (Good)'},
            {'color':'#50ccaa','title':'2 (Fair)'},
            {'color':'#f0e641','title':'3 (Moderate)'},
            {'color':'#ff5050','title':'4 (Poor)'},
            {'color':'#960032','title':'5 (Very Poor)'},
            {'color':'#7d2181','title':'6 (Extremely Poor)'},
        ]
    },
}

# --- API Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metadata Loading
def load_metadata_and_times():
    """
    Loads metadata, creates a time-to-COG mapping, and finds the start/end times.
    """
    if not os.path.exists(METADATA_FILE_PATH):
        raise FileNotFoundError(f"Metadata file not found at: {METADATA_FILE_PATH}")
    
    with open(METADATA_FILE_PATH, 'r') as f:
        metadata = json.load(f)

    time_mapping = {}
    all_times = []
    for entry in metadata:
        try:
            dt_obj = datetime.strptime(entry['timestep'], "%Y-%m-%d %H:%M")
        except (ValueError, KeyError):
            dt_obj = datetime.strptime(entry['timestep'], "%Y-%m-%d %H:%M:%S")
        
        time_iso = dt_obj.isoformat() + "Z"
        all_times.append(dt_obj)
        time_mapping[time_iso] = entry["files"]
    
    start_time = min(all_times).isoformat() + "Z"
    stop_time = max(all_times).isoformat() + "Z"
    
    return time_mapping, start_time, stop_time

# Precalc COG mapping and color ramp mapping on page load
TIME_COG_MAPPING, START_TIME, STOP_TIME = load_metadata_and_times()

# --- Catalog Generation Logic ---
def create_catalog_json():
    
    wms_members = []

    for pollutant, legend_def in POLLUTANT_LEGENDS.items():
        pollutant_upper = pollutant.upper()
        layer_item = {
            "name": pollutant_upper,
            "type": "wms",
            "url": f"{SELF_URL}/wms_proxy",
            "layers": "TiTiler WMS",
            "description": f"Hourly air quality maps for {pollutant_upper}.",
            "rectangle": {
                "west": 23.209, "south": 42.599, "east": 23.434, "north": 42.761
            },
            "styles": pollutant_upper,
            "legends": [legend_def],
            "startTime": START_TIME,
            "stopTime": STOP_TIME,
            "currentTime": START_TIME,
            "timeSeriesCanSmooth": True
        }
        wms_members.append(layer_item)

    # Create the final catalog structure with a group (already be a group?)
    terria_catalog = {
        "catalog": [
            {
            "name": "Air Quality Maps",
            "type": "group",
            "members": wms_members
            }
        ]
        }
    return terria_catalog

# --- API Endpoints ---
@app.get("/catalog.json")
async def get_catalog():
    try:
        catalog_data = create_catalog_json()
        return JSONResponse(content=catalog_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/wms_proxy")
async def wms_proxy(request: Request):
    query_params = dict(request.query_params)
    print(f"\n--- PROXY INCOMING REQUEST ---")
    print(f"URL: {request.url}")
    print(f"Received params: {query_params}")

    request_type = query_params.get("request", "").lower()

    # TilerFactory uses a single endpoint for all WMS requests.
    titiler_wms_base_url = f"{TITILER_ENDPOINT}/bands/wms"

    if request_type == "getcapabilities":
        print("\n--- Handling: GetCapabilities request ---")
        
        # Provide one representativeto generate base capabilities
        first_time = next(iter(TIME_COG_MAPPING.keys()))
        first_cog_files = TIME_COG_MAPPING[first_time]
        # Pick one pollutant (e.g., 'o3') to represent the dataset structure
        cog_path = first_cog_files["o3"]
        
        filename = os.path.basename(cog_path)
        cog_url = f"{BASE_COG_URL.rstrip('/')}/{filename}"
        
        titiler_params = {
            'request': 'GetCapabilities',
            'service': 'WMS',
            'version': '1.3.0',
            'layers': cog_url
        }
        
        print(f'Sent params: {titiler_params}')

        # Get capabilties xml so I can interject with time and style options
        async with httpx.AsyncClient() as client:
            try:
                # Get capabilities XML
                req = client.build_request("GET", titiler_wms_base_url, params=titiler_params)
                print(f"--- Full Request URL Sent to TiTiler ---\n{req.url}\n")
                resp = await client.send(req)
                print(f"Titiler Response Status Code: {resp.status_code}\n")
                resp.raise_for_status()
                xml_content = await resp.aread()
                root = ET.fromstring(xml_content)
                layer_element = root.find(f".//Layer[Name='{cog_url}']")

                if layer_element is not None:
                    # Rename the layer to what Terria expects ("bands")
                    name_tag = layer_element.find("Name")
                    if name_tag is not None:
                        name_tag.text = "TiTiler WMS"
                    title_tag = layer_element.find("Title")
                    if title_tag is not None:
                        title_tag.text = "TiTiler WMS"

                    # Create the time Dimension tag
                    time_dimension = ET.Element("Dimension")
                    time_dimension.set("name", "time")
                    time_dimension.set("units", "ISO8601")
                    time_dimension.set("default", START_TIME)
                    time_dimension.text = f"{START_TIME}/{STOP_TIME}/PT1H"
                    layer_element.append(time_dimension)
                    print("Successfully injected Time Dimension into GetCapabilities XML.")
                    # Interject style dimensions
                    pollutants = ["O3", "NO2", "SO2", "PM10", "PM25", "EAQI", "DOMINANT"]
                    for p in pollutants:
                        style_el = ET.Element("Style")
                        name_el = ET.Element("Name")
                        name_el.text = p
                        style_el.append(name_el)
                        title_el = ET.Element("Title")
                        title_el.text = f"Pollutant: {p}"
                        style_el.append(title_el)
                        abstract_el = ET.Element("Abstract")
                        abstract_el.text = f"Air quality measurement for {p}"
                        style_el.append(abstract_el)
                        layer_element.append(style_el)
                    print("Successfully injected Style definitions into GetCapabilities XML.")
                else:
                    print(f"Warning: Could not find Layer '{cog_url}' in TiTiler's GetCapabilities response.")

                # Convert the modified XML tree back to a string
                modified_xml_string = ET.tostring(root, encoding='unicode')
                
                return Response(content=modified_xml_string, media_type="text/xml")

            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"Error contacting TiTiler service: {e}")
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=f"TiTiler error: {e.response.text}")

    elif request_type == "getmap":
        print("Handling: GetMap request")
        # Create a stable, hashable key from the query parameters
        cache_key = frozenset(request.query_params.items())

        # 1. CHECK CACHE (CACHE HIT)
        if cache_key in tile_cache:
            print("--- Handling: GetMap request (CACHE HIT) ---")
            cached_response = tile_cache[cache_key]
            # Return the cached content and headers directly
            return Response(
                content=cached_response["content"],
                media_type=cached_response["media_type"],
                status_code=cached_response["status_code"],
            )
        
        # Get maps based on time
        time = query_params.get("time", START_TIME)
        requested_dt = datetime.fromisoformat(time.replace("Z", "+00:00"))
        closest_time_iso = min(
            TIME_COG_MAPPING.keys(),
            key=lambda t: abs(requested_dt - datetime.fromisoformat(t.replace("Z", "+00:00")))
        )
        print(f'Using time: {time}')
        cog_files_for_timestep = TIME_COG_MAPPING[closest_time_iso]
        # Get map based on time + style(measurement)
        requested_style = query_params.get("styles", "O3").split(',')[0].lower()
        if requested_style not in cog_files_for_timestep:
            raise HTTPException(status_code=404, detail=f"Invalid style/pollutant: '{requested_style}'")
        print(f"Using style: '{requested_style}'")
        cog_path = cog_files_for_timestep[requested_style]
        filename = os.path.basename(cog_path)
        cog_url = f"{BASE_COG_URL.rstrip('/')}/{filename}"

        titiler_wms_url = f"{TITILER_ENDPOINT}/bands/wms"

        # Finalize getmap params
        titiler_params = dict(request.query_params)
        titiler_params['layers'] = cog_url
        titiler_params.pop('time', None)
        titiler_params.pop('styles', None)
        limits = EAQI_LIMITS[requested_style]
        titiler_params["rescale"] = f"0,65535"
        titiler_params["colormap_name"] = requested_style

        print(f"Forwarding to Titiler URL: {titiler_wms_url}")
        print(f'Params: {titiler_params.keys()}')

        # Make request with POST
        async with httpx.AsyncClient() as client:
            try:
                req = client.build_request(
                    "GET",
                    titiler_wms_url,
                    params=titiler_params)
                print(f"--- Full Request URL Sent to TiTiler ---\n{req.url}\n")
                resp = await client.send(req)
                resp.raise_for_status()
                # Read the response content to be able to cache it
                response_content = await resp.aread()

                # 3. STORE THE NEW TILE IN THE CACHE
                tile_cache[cache_key] = {
                    "content": response_content,
                    "media_type": resp.headers.get("Content-Type"),
                    "status_code": resp.status_code,
                }
                print(f"--- Stored new tile in cache. Cache size: {len(tile_cache)}/{tile_cache.maxsize} ---")

                # Return the response to the user
                return Response(
                    content=response_content,
                    media_type=resp.headers.get("Content-Type"),
                    status_code=resp.status_code,
                )
            except httpx.RequestError as e:
                print(f"HTTPX Request Error: {e}")
                raise HTTPException(status_code=502, detail=f"Error contacting TiTiler service: {e}")
            except httpx.HTTPStatusError as e:
                print(f"HTTPX Status Error: {e.response.text}")
                raise HTTPException(status_code=e.response.status_code, detail=f"TiTiler error: {e.response.text}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise HTTPException(status_code=500, detail="An unexpected error occurred in the WMS proxy.")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported WMS request type: '{request_type}'")

  


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)