# the purpose of this code is to query Google Maps API to get the congestion travel times
# this code is similar to setting_weights_v1.py, but it includes the congestion travel times AND it improves on the gmaps queries

# general packages
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from geopy.distance import geodesic
from datetime import datetime
from itertools import product
import re
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm

# for Google Maps API
import urllib.parse
import requests
import time
from typing import List, Dict, Tuple

# for network graph and routing
import networkx as nx
import ast

# -----------------
# (1) Prepare data for Google Maps API Request
# -----------------

# -----------------
# (1a) get API key

# I saved the path to the API key in a text file so that I can reference it as part of the project, 
# but not need to duplicate the actual key
def read_path_to_google_maps_api_key():
    read_read_path = os.path.join(os.path.dirname(os.getcwd()), 'api_keys', 'path_to_google_maps_api_key_bp.txt')
    with open(read_read_path, 'r') as file:
        read_path = file.read()
    return read_path

def read_google_maps_api_key(read_path):
    with open(read_path, 'r') as file:
        api_key = file.read()
    return api_key

# I'll want to use the distance-matrix API
# https://developers.google.com/maps/documentation/distance-matrix/overview
# google maps offers a $200 monthly credit
# https://mapsplatform.google.com/pricing/#pricing-grid
# I'm charged $5 per 1000 elements for the basic distance matrix API
# https://developers.google.com/maps/documentation/distance-matrix/usage-and-billing
# so technically, I could make 40,000 requests per month for free
# where I can manage billing
# https://console.cloud.google.com/billing

# Unfortunately Google Distance Matrix returns a crossjoin of all origin and destination pairs
# so I will be forced to run a separate query for each of the tens of thousands of pairs
# otherwise I would be charged for all the elements in the crossjoin

# -----------------
# (1b) get matrix I want to compare

def read_od_matrix_path():
    read_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'od_matrix.csv')
    return read_path

def read_od_matrix(od_matrix_path):
    od_matrix = pd.read_csv(od_matrix_path)
    od_matrix['origin_node'] = od_matrix['origin_node'].astype(str)
    od_matrix['destination_node'] = od_matrix['destination_node'].astype(str)
    return od_matrix

def read_nodes_path():
    nodes_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'nodes.gpkg')
    return nodes_path

def read_nodes(nodes_path):
    nodes = gpd.read_file(nodes_path)
    return nodes

def add_lat_long_to_od_matrix(od_matrix, nodes_df):
    od_matrix = od_matrix.merge(nodes_df[['Node', 'x', 'y']], left_on='origin_node', right_on='Node', how='left')
    od_matrix = od_matrix.rename(columns={'y': 'origin_latitude', 'x': 'origin_longitude'}).drop(columns='Node')
    od_matrix = od_matrix.merge(nodes_df[['Node', 'x', 'y']], left_on='destination_node', right_on='Node', how='left')
    od_matrix = od_matrix.rename(columns={'y': 'destination_latitude', 'x': 'destination_longitude'}).drop(columns='Node')
    return od_matrix

# calculating distances is important because I don't want to waste API queries on short trips. I learn much more on longer trips
# note: this is distance as the crow flies so it will be different from the google maps distance
def add_distance_in_miles(od_matrix):
    # (1) Create empty list to store distances
    distances = []
    # (2) Loop through each row and calculate geodesic distance between origin and destination
    for idx, row in od_matrix.iterrows():
        origin = (row['origin_latitude'], row['origin_longitude'])
        destination = (row['destination_latitude'], row['destination_longitude'])
        # (3) Calculate distance in miles using geopy's geodesic
        distance = geodesic(origin, destination).miles
        distances.append(distance)
    # (4) Add distances as new column
    od_matrix['distance_in_miles'] = distances
    return od_matrix

def filter_distance_in_miles(od_matrix, min_distance=2):
    od_matrix = od_matrix[od_matrix['distance_in_miles'] >= min_distance]
    return od_matrix

# because there are ~134k unique node pairs and I am price constrained in my query, I need an appropriate sampling method
# I am going to try to sample unique census block group pairs. This will remove the short trip node pairs, but what I'm really
# concerned about in this analysis are people who have to travel a decent distance to get to work
# note: the short distance workers are important because they affect the congestion on the roadways
# there are 63k unique census block group pairs
# I can pick the 10k most common of those census block group pairs to use in my analysis

def pick_most_common_cbg_pairs(od_matrix, sample=10_000):
    # (1) create a column for destination CBG
    od_matrix['census_block_group_destination'] = od_matrix['w_geocode'].astype('str').str[:12]
    # (2) group by origin CBG and destination CBG and count
    od_matrix_grouped = od_matrix.groupby(['census_block_group_origin', 'census_block_group_destination']).size().reset_index(name='count')
    od_matrix_grouped = od_matrix_grouped.sort_values(by='count', ascending=False)
    # (3) pick the most common
    od_matrix_grouped = od_matrix_grouped.head(sample)
    return od_matrix_grouped

def pick_random_sample_cbg_pairs(od_matrix, od_matrix_grouped):
    # (1) filter od_matrix for the od_matrix_grouped origin and destination CBGs
    od_matrix_grouped = od_matrix_grouped[['census_block_group_origin', 'census_block_group_destination']]
    od_matrix_filtered = od_matrix.merge(od_matrix_grouped, on=['census_block_group_origin', 'census_block_group_destination'], how='inner')
    # (2) for each origin and destination CBG, pick a random sample
    od_matrix_sample = od_matrix_filtered.groupby(['census_block_group_origin', 'census_block_group_destination']).apply(
        lambda x: x.sample(n=min(len(x), 1), random_state=42)
    ).reset_index(drop=True)
    return od_matrix_sample

# prepare 5min_designation as actual time intervals

def capture_base_time(departure_time):
    # Match anything between "leave_" and "am_to_"
    pattern = r"leave_(\d+)am_to_"
    match = re.search(pattern, departure_time)
    return match.group(1) if match else None

def capture_interval(departure_time):
    # match after 'interval_'
    pattern = r"interval_(\d+)"
    match = re.search(pattern, departure_time)
    return match.group(1) if match else None

def min_designation(column):
    # match number before 'min_designation'
    pattern = r"(\d+)min_designation"
    match = re.search(pattern, column)
    return match.group(1) if match else None

def create_new_departure_time(base_time, interval, minute_designation):
    # Convert base_time to string to handle length checking
    base_str = str(base_time)
    interval = int(interval)
    # Handle single digit input (e.g., '9' becomes '09:00')
    if len(base_str) == 1:
        time_str = f"0{base_str}:00"
    # Handle three/four digit input (e.g., '630' becomes '06:30')
    else:
        # Extract hours and minutes
        if len(base_str) == 3:
            hours = base_str[0]
            minutes = base_str[1:]
        else:  # len == 4
            hours = base_str[:2]
            minutes = base_str[2:]
        time_str = f"{hours.zfill(2)}:{minutes}"
    # Convert string to datetime object
    base_datetime = datetime.strptime(time_str, "%H:%M")
    minutes_to_add = (interval - 1) * minute_designation
    # Add minutes to base time
    new_time = base_datetime + timedelta(minutes=minutes_to_add)
    # Return formatted time string
    return new_time.strftime("%H:%M:%S")
    # return new_time.strftime("%H:%M")

def prepare_departure_time_for_od_matrix(od_matrix, column ='5min_designation'):
    # extract minute designation
    minute_designation = int(min_designation(column))
    od_matrix['base_time'] = od_matrix[column].apply(capture_base_time)
    od_matrix['interval'] = od_matrix[column].apply(capture_interval)
    od_matrix['departure_time'] = od_matrix.apply(
        lambda row: create_new_departure_time(row['base_time'], row['interval'], minute_designation), 
        axis=1
    )
    return od_matrix

# -----------------
# (2) Run Google Maps API Request
# -----------------

def convert_hst_to_utc_epoch(date_str, time_str):
    # Create timezone objects
    hst = pytz.timezone('Pacific/Honolulu')  # HST timezone
    utc = pytz.UTC
    # Combine date and time string
    datetime_str = f"{date_str} {time_str}"
    # Create datetime object in HST
    hst_dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    hst_dt = hst.localize(hst_dt)
    # Convert to UTC
    utc_dt = hst_dt.astimezone(utc)
    # Convert to epoch
    epoch = int(utc_dt.timestamp())
    return epoch

# this is the same code but where I can set a departure time (i.e. model congestion)
def create_distance_matrix_url_set_departure(api_key, origin_latitude, origin_longitude, 
                             destination_latitude, destination_longitude, 
                             departure_time, date="2025-06-02"):
    # 1 Create a Google Maps Distance Matrix API URL for a single origin-destination pair
    # 2 All coordinates should be floats, departure_time in HH:MM format, date in YYYY-MM-DD
    # 3 Returns formatted URL string for the Distance Matrix API request
    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    
    # 4 Format coordinates as string pairs
    origin = f"{origin_latitude},{origin_longitude}"
    destination = f"{destination_latitude},{destination_longitude}"
    
    # 5 Build base parameters dict
    params = {
        'origins': origin,
        'destinations': destination,
        'key': api_key,
        'mode': 'driving'
        # 'traffic_model': 'pessimistic' # trying this out
    }
    
    # 6 Handle departure time - always use epoch since we have default date
    # datetime_str = f"{date} {departure_time}"
    # epoch = int(datetime.strptime(datetime_str, "%Y-%m-%d %H:%M").timestamp())
    # params['departure_time'] = epoch
    epoch = convert_hst_to_utc_epoch(date, departure_time)
    params['departure_time'] = epoch
    
    # 7 Construct final URL with encoded parameters
    query_string = urllib.parse.urlencode(params)
    final_url = f"{base_url}?{query_string}"
    return final_url

def apply_maps_api_urls_to_od_matrix_sample_congestion(od_matrix_sample, api_key):
    # (1) Create empty list to store URLs
    urls = []
    # (2) Loop through each row and create URL for origin-destination pair
    for idx, row in od_matrix_sample.iterrows():
        url = create_distance_matrix_url_set_departure(api_key, row['origin_latitude'], row['origin_longitude'], 
                             row['destination_latitude'], row['destination_longitude'], row['departure_time'])
        urls.append(url)
    # (3) Add URLs as new column
    od_matrix_sample['congestion_api_url'] = urls
    return od_matrix_sample

def pull_json_data_single_url(url):
    # (1) Send GET request to URL
    response = requests.get(url)
    # (2) Extract JSON data
    try:
        data = response.json()
        duration_in_traffic_seconds = data['rows'][0]['elements'][0]['duration_in_traffic']['value']
        return duration_in_traffic_seconds
    except (KeyError, IndexError) as e:
        print(f"Error extracting duration_in_traffic: {e}")
        return None

def loop_through_od_matrix_sample_and_pull_data_from_urls(od_matrix_sample):
    # Create an empty list to store all durations
    durations = []
    # Create a progress bar
    for i, url in enumerate(tqdm(od_matrix_sample['congestion_api_url'], 
                                desc="Fetching traffic durations", 
                                unit="request")):
        duration_in_traffic_seconds = pull_json_data_single_url(url)
        durations.append(duration_in_traffic_seconds)
        # Add delay every 50 requests
        if (i + 1) % 50 == 0:
            time.sleep(1)  # Sleep for 1 second
    # Add the collected durations as a new column
    od_matrix_sample['duration_in_traffic_seconds'] = durations
    return od_matrix_sample

# -----------------
# (3) Main function to run the Google Maps API query
# -----------------

def main_free_flow_query(min_distance = 2, sample=10_000):
    # (1) Read in API key
    read_path = read_path_to_google_maps_api_key()
    api_key = read_google_maps_api_key(read_path)
    
    # (2) Read in OD matrix
    od_matrix_path = read_od_matrix_path()
    od_matrix = read_od_matrix(od_matrix_path)
    
    # (3) Read in nodes
    nodes_path = read_nodes_path()
    nodes_df = read_nodes(nodes_path)
    
    # (4) Add lat/long to OD matrix
    od_matrix = add_lat_long_to_od_matrix(od_matrix, nodes_df)
    
    # (5) Add distance in miles to OD matrix
    od_matrix = add_distance_in_miles(od_matrix)
    
    # (6) Filter out short trips
    od_matrix = filter_distance_in_miles(od_matrix, min_distance = min_distance)
    
    # (7) Pick most common CBG pairs
    od_matrix_grouped = pick_most_common_cbg_pairs(od_matrix, sample=sample)
    
    # (8) Pick random sample of CBG pairs
    od_matrix_sample = pick_random_sample_cbg_pairs(od_matrix, od_matrix_grouped)

    # (9) Prepare departure time for OD matrix
    od_matrix_sample = prepare_departure_time_for_od_matrix(od_matrix_sample)
    # od_matrix_sample = od_matrix_sample[od_matrix_sample['person_id'] == 119543] # will delete this later
    
    # (10) Apply Google Maps API URLs to OD matrix
    od_matrix_sample = apply_maps_api_urls_to_od_matrix_sample_congestion(od_matrix_sample, api_key)

    # (11) Loop through OD matrix sample and pull data from URLs
    od_matrix_sample = loop_through_od_matrix_sample_and_pull_data_from_urls(od_matrix_sample)

    # (12) drop free_flow_api_url (I don't want to share my api key)
    # od_matrix_sample = od_matrix_sample.drop(columns='free_flow_api_url')
    
    return od_matrix_sample

# -----------------
# (4) Compare Simulation with Google Maps
# -----------------

def read_simulation_results(file_name):
    path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'congestion_iterations', file_name)
    df = pd.read_csv(path, dtype={'person_id': str})
    return df

# def read_google_maps_congestion_results():
#     path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'gmaps_congestion_route_times.csv')
#     df = pd.read_csv(path, dtype={'person_id': str})
#     return df

def read_google_maps_congestion_results():
    path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'gmaps_congestion_route_times.csv')
    
    dtypes = {
        'person_id': str,
        'duration_in_traffic_seconds': 'int32'
    }
    
    # Use parse_dates with the correct time format (%H:%M:%S for 24-hour time)
    df = pd.read_csv(
        path,
        dtype=dtypes,
        parse_dates=['departure_time'],
        date_parser=lambda x: pd.to_datetime(x, format='%H:%M:%S')
    )
    
    df['gmaps_completion_time'] = df['departure_time'] + pd.to_timedelta(df['duration_in_traffic_seconds'], unit='seconds')
    
    return df

def compare_simulation_with_google_maps(simulation_results, google_maps_results):
    # (1) rename 'completion_time' to 'simulation_completion_time'
    simulation_results = simulation_results.rename(columns={'completion_time': 'simulation_completion_time'})
    # (2) calculate 'gmaps_completion_time' as 'duration_in_traffic_seconds' + 'departure_time'
    # google_maps_results['gmaps_completion_time'] = google_maps_results['departure_time'] + google_maps_results['duration_in_traffic_seconds']
    # (3) merge simulation results with google maps results
    merged_df = pd.merge(google_maps_results, simulation_results, on='person_id', how='inner')
    # (4) calculate the difference between the two
    # merged_df['difference'] = merged_df['simulation_completion_time'] - merged_df['gmaps_completion_time']
    return merged_df

