# the purpose of this code is to set correct weights for the different connection types
# I'm going to use Google Maps API to help with this
# first I will start with free flow travel time to get from Node1 to Node2
# later, I will see how Google Maps API travel time during congestion hours compares to my model

# for Google Maps API
import nest_asyncio
nest_asyncio.apply()

# general packages
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from geopy.distance import geodesic
from datetime import datetime
from itertools import product

# for Google Maps API
import urllib.parse
import requests
import asyncio
import aiohttp
import time
from typing import List, Dict, Tuple

# for network graph and routing
import networkx as nx
import ast

# parallel processing
from concurrent.futures import ProcessPoolExecutor

# -----------------
# (1) Prepare data for Google Maps API Request
# -----------------

# -----------------
# (1a) get API key

# I saved the path to the API key in a text file so that I can reference it as part of the project, 
# but not need to duplicate the actual key
def read_path_to_google_maps_api_key():
    read_read_path = os.path.join(os.path.dirname(os.getcwd()), 'api_keys', 'path_to_google_maps_api_key.txt')
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

# -----------------
# (2) Run Google Maps API Request
# -----------------

# -----------------
# (2a) create URLs for single Origin / Destination pairs

def create_distance_matrix_url_free_flow(api_key, origin_latitude, origin_longitude, 
                             destination_latitude, destination_longitude):
    # 1 Create a Google Maps Distance Matrix API URL for a single origin-destination pair
    # 2 All coordinates should be floats
    # 3 Returns formatted URL string for the Distance Matrix API request
    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    
    # 4 Format coordinates as string pairs
    origin = f"{origin_latitude},{origin_longitude}"
    destination = f"{destination_latitude},{destination_longitude}"
    
    # 5 Build base parameters dict
    params = {
        'origins': origin,
        'destinations': destination,
        'key': api_key
    }
    
    # 6 Construct final URL with encoded parameters
    query_string = urllib.parse.urlencode(params)
    final_url = f"{base_url}?{query_string}"
    return final_url

# this is the same code but where I can set a departure time (i.e. model congestion)
def create_distance_matrix_url_set_departure(api_key, origin_latitude, origin_longitude, 
                             destination_latitude, destination_longitude, 
                             departure_time="03:00", date="2025-07-01"):
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
    }
    
    # 6 Handle departure time - always use epoch since we have default date
    datetime_str = f"{date} {departure_time}"
    epoch = int(datetime.strptime(datetime_str, "%Y-%m-%d %H:%M").timestamp())
    params['departure_time'] = epoch
    
    # 7 Construct final URL with encoded parameters
    query_string = urllib.parse.urlencode(params)
    final_url = f"{base_url}?{query_string}"
    return final_url

def apply_maps_api_urls_to_od_matrix_sample_free_flow(od_matrix_sample, api_key):
    # (1) Create empty list to store URLs
    urls = []
    # (2) Loop through each row and create URL for origin-destination pair
    for idx, row in od_matrix_sample.iterrows():
        url = create_distance_matrix_url_free_flow(api_key, row['origin_latitude'], row['origin_longitude'], 
                             row['destination_latitude'], row['destination_longitude'])
        urls.append(url)
    # (3) Add URLs as new column
    od_matrix_sample['free_flow_api_url'] = urls
    return od_matrix_sample

def apply_maps_api_urls_to_od_matrix_sample_congestion(od_matrix_sample, api_key):
    # (1) Create empty list to store URLs
    urls = []
    # (2) Loop through each row and create URL for origin-destination pair
    for idx, row in od_matrix_sample.iterrows():
        url = create_distance_matrix_url_set_departure(api_key, row['origin_latitude'], row['origin_longitude'], 
                             row['destination_latitude'], row['destination_longitude'])
        urls.append(url)
    # (3) Add URLs as new column
    od_matrix_sample['congestion_api_url'] = urls
    return od_matrix_sample

# -----------------
# (2b) batch async requests to Google Maps API

# Google did not like this. When I query for the congestion times, I should
# (1) create a Google Cloud Account with Berkshire email
# (2) create a project
# (3) include stringent rate limits, run linearly rather than async. Include rest periods

# async def process_urls_batch(df: pd.DataFrame, url_column_name: str, max_requests_per_minute: int = 60) -> List[dict]:
#     # (2) Calculate delay between requests
#     delay = 60.0 / max_requests_per_minute
#     results = []
#     # (3) Create async session
#     async with aiohttp.ClientSession() as session:
#         # (4) Process each URL
#         for i, url in enumerate(df[url_column_name]):
#             try:
#                 # (5) Make API request
#                 async with session.get(url) as response:
#                     data = await response.json()
#                     if data['status'] == 'OK':
#                         element = data['rows'][0]['elements'][0]  # Since it's 1-to-1 mapping
#                         results.append({
#                             'distance_meters': element['distance']['value'],
#                             'distance_text': element['distance']['text'],
#                             'duration_seconds': element['duration']['value'],
#                             'duration_text': element['duration']['text'],
#                             'status': element['status']
#                         })
#                     else:
#                         print(f"Request failed with status: {data['status']} for URL index {i}")
#                         results.append({
#                             'distance_meters': None,
#                             'distance_text': None,
#                             'duration_seconds': None,
#                             'duration_text': None,
#                             'status': data['status']
#                         })
#             except Exception as e:
#                 print(f"Error processing URL {i}: {str(e)}")
#                 results.append({
#                     'distance_meters': None,
#                     'distance_text': None,
#                     'duration_seconds': None,
#                     'duration_text': None,
#                     'status': 'ERROR'
#                 })
#             # (6) Wait before next request
#             await asyncio.sleep(delay)
#     return results

# async def batch_process_api_requests(df: pd.DataFrame, url_column_name: str = 'free_flow_api_url', max_requests_per_minute: int = 60) -> pd.DataFrame:
#     # (7) Run async processing and return combined DataFrame
#     results = await process_urls_batch(df, url_column_name, max_requests_per_minute)
#     results_df = pd.DataFrame(results)
#     return pd.concat([df, results_df], axis=1)

# -----------------
# (3) Main function to run the Google Maps API query
# -----------------

def main_free_flow_query(min_distance = 2, sample=10_000, max_requests_per_minute=5_000):
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
    
    # (9) Apply Google Maps API URLs to OD matrix
    od_matrix_sample = apply_maps_api_urls_to_od_matrix_sample_free_flow(od_matrix_sample, api_key)
    print(f"number of rows: {len(od_matrix_sample)}")
    
    # (10) Run batch processing of API requests
    results_df = asyncio.run(batch_process_api_requests(od_matrix_sample, max_requests_per_minute=max_requests_per_minute))

    # (11) drop free_flow_api_url (I don't want to share my api key)
    results_df = results_df.drop(columns='free_flow_api_url')
    
    return results_df

# -----------------
# (4) Create vehicles_df (functions come from route_optimization_v2.py)
# -----------------

# this function is already created
# def read_od_matrix_path():
#     read_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'od_matrix.csv')
#     return read_path

# def read_od_matrix(od_matrix_path):
#     od_matrix = pd.read_csv(od_matrix_path)
#     od_matrix['origin_node'] = od_matrix['origin_node'].astype(str)
#     od_matrix['destination_node'] = od_matrix['destination_node'].astype(str)
#     return od_matrix

def read_edges_path():
    edges_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'edges.gpkg')
    return edges_path

def read_edges(edges_path):
    edges = gpd.read_file(edges_path)
    return edges

def create_network_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
    #(1) Create directed graph from edges dataframe using free_flow_travel_time as weight
    G = nx.DiGraph()
    #(2) Add edges with their attributes
    for _, row in edges_df.iterrows():
        G.add_edge(
            row['Node1'],
            row['Node2'],
            weight=row['free_flow_travel_time'],
            length=row['length'], # in meters
            length_miles=row['length_miles'],
            FFS=row['FFS'],
            capacity=row['capacity'],
            edge_category=row['edge_category'],
            link_id=row['link_id']
        )
    #(3) Basic validation
    print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

# this code was incorrect. It returned disjointed routes
# def calculate_initial_routes(G: nx.DiGraph, od_matrix: pd.DataFrame, edges_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, List[int]], pd.DataFrame]:
#     #(1) Verify all OD nodes are truly on two-way streets
#     print("\nVerifying OD matrix nodes:")
#     all_od_nodes = pd.concat([od_matrix['origin_node'], od_matrix['destination_node']]).unique()
#     print(f"Total unique nodes in OD matrix: {len(all_od_nodes)}")
#     for node in all_od_nodes:
#         node = str(node)  # ensure string type
#         # Count occurrences as Node1 and Node2
#         as_node1 = edges_df[edges_df['Node1'] == node].shape[0]
#         as_node2 = edges_df[edges_df['Node2'] == node].shape[0]
#         if as_node1 == 0 or as_node2 == 0:
#             print(f"Node {node} appears as Node1: {as_node1} times, as Node2: {as_node2} times")
#             # Show the actual edges for this node
#             print("Edges:")
#             print(edges_df[edges_df['Node1'] == node][['Node1', 'Node2', 'oneway', 'link_id']])
#             print(edges_df[edges_df['Node2'] == node][['Node1', 'Node2', 'oneway', 'link_id']])
#             break
#     #(2) Create node pair to link_id mapping
#     link_lookup = {(str(row['Node1']), str(row['Node2'])): row['link_id'] for _, row in edges_df.iterrows()}
#     #(3) Initialize output dataframe
#     vehicles_df = od_matrix.copy()
#     vehicles_df['current_link'] = None
#     vehicles_df['route_links'] = None
#     vehicles_df['routing_status'] = 'pending'
#     #(4) Calculate shortest paths for unique OD pairs
#     route_dict = {}
#     unique_od_pairs = od_matrix.groupby(['origin_node', 'destination_node']).size().reset_index()
#     print(f"Calculating routes for {len(unique_od_pairs)} unique OD pairs...")
#     #(5) Track statistics and failures
#     successful_routes = 0
#     failed_routes = 0
#     failures_list = []
#     unreachable_pairs = set()
#     for _, row in unique_od_pairs.iterrows():
#         origin = str(row['origin_node'])
#         destination = str(row['destination_node'])
#         if (origin, destination) in unreachable_pairs:
#             continue
#         try:
#             if origin not in G or destination not in G:
#                 failures_list.append({
#                     'origin': origin,
#                     'destination': destination,
#                     'failure_type': 'node_not_in_network',
#                     'origin_in_network': origin in G,
#                     'destination_in_network': destination in G
#                 })
#                 unreachable_pairs.add((origin, destination))
#                 continue
#             path_nodes = nx.shortest_path(G, origin, destination, weight='weight')
#             #(6) Convert to link IDs
#             path_links = []
#             for i in range(len(path_nodes) - 1):
#                 node_pair = (path_nodes[i], path_nodes[i + 1])
#                 if node_pair not in link_lookup:
#                     print(f"Warning: Missing link between nodes {node_pair}")
#                     continue
#                 link_id = link_lookup[node_pair]
#                 path_links.append(link_id)
#             #(7) Only process if we found a valid path
#             if path_links:
#                 mask = (od_matrix['origin_node'] == row['origin_node']) & (od_matrix['destination_node'] == row['destination_node'])
#                 person_ids = od_matrix.loc[mask, 'person_id']
#                 for pid in person_ids:
#                     route_dict[pid] = path_links
#                     vehicles_df.loc[vehicles_df['person_id'] == pid, 'route_links'] = str(path_links)
#                     vehicles_df.loc[vehicles_df['person_id'] == pid, 'current_link'] = path_links[0]
#                     vehicles_df.loc[vehicles_df['person_id'] == pid, 'routing_status'] = 'success'
#                     successful_routes += 1
#             else:
#                 failures_list.append({
#                     'origin': origin,
#                     'destination': destination,
#                     'failure_type': 'no_valid_links'
#                 })
#                 unreachable_pairs.add((origin, destination))
#                 failed_routes += 1
#         except nx.NetworkXNoPath:
#             failures_list.append({
#                 'origin': origin,
#                 'destination': destination,
#                 'failure_type': 'no_path_found'
#             })
#             unreachable_pairs.add((origin, destination))
#             failed_routes += 1
#             continue
#     #(8) Create failures DataFrame
#     failures_df = pd.DataFrame(failures_list)
#     #(9) Print enhanced summary statistics
#     print("\nRouting Summary:")
#     print(f"Successfully routed: {successful_routes} vehicles")
#     print(f"Failed to route: {failed_routes} OD pairs")
#     print(f"Total unreachable OD pairs: {len(unreachable_pairs)}")
#     if not failures_df.empty:
#         print("\nFailure Analysis:")
#         print(f"Number of unique failing origin nodes: {failures_df['origin'].nunique()}")
#         print(f"Number of unique failing destination nodes: {failures_df['destination'].nunique()}")
#         print("\nTop 5 problematic origin nodes:")
#         origin_counts = failures_df['origin'].value_counts().head()
#         print(origin_counts)
#         print("\nTop 5 problematic destination nodes:")
#         dest_counts = failures_df['destination'].value_counts().head()
#         print(dest_counts)
#     #(10) Mark vehicles with no route
#     mask = vehicles_df['routing_status'] == 'pending'
#     vehicles_df.loc[mask, 'routing_status'] = 'failed'
#     unrouted = vehicles_df['route_links'].isna().sum()
#     if unrouted > 0:
#         print(f"Total vehicles without routes: {unrouted}")
#     return vehicles_df, failures_df

def calculate_node_paths(G: nx.DiGraph, od_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the node-to-node paths for each origin-destination pair.
    Returns the od_matrix with an additional 'node_path' column.
    """
    # Initialize output dataframe
    vehicles_df = od_matrix.copy()
    vehicles_df['node_path'] = None
    vehicles_df['routing_status'] = 'pending'
    
    # Calculate paths for unique OD pairs
    unique_od_pairs = od_matrix.groupby(['origin_node', 'destination_node']).size().reset_index()
    print(f"Calculating paths for {len(unique_od_pairs)} unique OD pairs...")
    
    for _, row in unique_od_pairs.iterrows():
        origin = str(row['origin_node'])
        destination = str(row['destination_node'])
        
        try:
            # Find shortest path
            path_nodes = nx.shortest_path(G, origin, destination, weight='weight')
            
            # Update all vehicles with this OD pair
            mask = ((vehicles_df['origin_node'] == row['origin_node']) & 
                   (vehicles_df['destination_node'] == row['destination_node']))
            vehicles_df.loc[mask, 'node_path'] = str(path_nodes)
            vehicles_df.loc[mask, 'routing_status'] = 'success'
            
        except nx.NetworkXNoPath:
            mask = ((vehicles_df['origin_node'] == row['origin_node']) & 
                   (vehicles_df['destination_node'] == row['destination_node']))
            vehicles_df.loc[mask, 'routing_status'] = 'no_path'
            continue
    
    # Print summary
    success_count = (vehicles_df['routing_status'] == 'success').sum()
    fail_count = (vehicles_df['routing_status'] != 'success').sum()
    print(f"\nPath Calculation Summary:")
    print(f"Successful paths: {success_count}")
    print(f"Failed paths: {fail_count}")
    
    return vehicles_df

def convert_node_paths_to_links(vehicles_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert node paths to link sequences using the edges dataframe.
    Returns the vehicles_df with an additional 'route_links' column.
    """
    # Create node pair to link_id lookup
    link_lookup = {(str(row['Node1']), str(row['Node2'])): row['link_id'] 
                  for _, row in edges_df.iterrows()}
    
    # Initialize route_links column
    vehicles_df['route_links'] = None
    vehicles_df['conversion_status'] = 'pending'
    
    # Process only vehicles with successful node paths
    mask = vehicles_df['routing_status'] == 'success'
    success_count = 0
    fail_count = 0
    
    for idx, row in vehicles_df[mask].iterrows():
        try:
            # Convert string representation of node path back to list
            node_path = eval(row['node_path'])  # Safe since we created this string ourselves
            
            # Convert to links
            path_links = []
            path_is_continuous = True
            
            for i in range(len(node_path) - 1):
                node_pair = (str(node_path[i]), str(node_path[i + 1]))
                if node_pair not in link_lookup:
                    path_is_continuous = False
                    print(f"Warning: Missing link between nodes {node_pair}")
                    break
                path_links.append(link_lookup[node_pair])
            
            if path_is_continuous and path_links:
                vehicles_df.loc[idx, 'route_links'] = str(path_links)
                vehicles_df.loc[idx, 'conversion_status'] = 'success'
                success_count += 1
            else:
                vehicles_df.loc[idx, 'conversion_status'] = 'discontinuous'
                fail_count += 1
                
        except Exception as e:
            vehicles_df.loc[idx, 'conversion_status'] = f'error: {str(e)}'
            fail_count += 1
    
    # Print summary
    print(f"\nLink Conversion Summary:")
    print(f"Successful conversions: {success_count}")
    print(f"Failed conversions: {fail_count}")
    
    return vehicles_df

def add_estimated_total_time(vehicles_df, edges_df):
    #1. create dictionary for quick lookup of free flow times
    free_flow_times = dict(zip(edges_df['link_id'], edges_df['free_flow_travel_time']))
    
    def calculate_route_time(route_links):
        #2. convert string to list if needed
        if isinstance(route_links, str):
            route_links = ast.literal_eval(route_links)
        #3. sum up times for valid route lists
        if isinstance(route_links, list):
            return sum(free_flow_times.get(link, 0) for link in route_links)
        return 0
    
    #4. apply the calculation to each row
    vehicles_df['estimated_total_time'] = vehicles_df['route_links'].apply(calculate_route_time)
    return vehicles_df

# if I don't set the edges_df and od_matrix, then the function will read them in
def main_create_vehicles_df(edges_df = None, od_matrix = None):
    # read the od_matrix and edges
    if edges_df is None:
        edges_path = read_edges_path()
        edges_df = read_edges(edges_path)
    if od_matrix is None:
        od_matrix_path = read_od_matrix_path()
        od_matrix = read_od_matrix(od_matrix_path)

    # filter the od_matrix based on the Google Maps routes
    gmaps_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'gmaps_free_flow_route_times.csv')
    gmaps_df = pd.read_csv(gmaps_path)
    od_matrix_sample = od_matrix[od_matrix['person_id'].isin(gmaps_df['person_id'])]

    # create the network graph
    G = create_network_graph(edges_df)
    # calculate routes using the pre-built graph
    # vehicles_df is another representation of the OD Matrix data
    vehicles_df, failures_df = calculate_initial_routes(G, od_matrix_sample, edges_df)
    if not failures_df.empty:
        print(f"Failures: {failures_df}")

    vehicles_df = add_estimated_total_time(vehicles_df, edges_df)
    return vehicles_df

# -----------------
# (5) Adjust weights for edges_df
# -----------------

# I used the following .py files to create an origin destination (demand) matrix, a nodes file, and an edges file
# origin_destination_v2.py
# road_link_v3.py
# route_optimization_v2.py

# from there I created a vehicles_df which matches the gmap output from steps 1-3 above
# vehicles_df was created in 241231_setting_weights_v1.ipynb

# my leading assumption when setting these weights is that the actual route that one takes between origin and destination
# is constant across the free flow travel time weights
# I think this can be a reasonable assumption because the free flow travel time weights are just multipliers for the 
# underlying speed limits of the roadways. These weights shouldn't somehow make a highway less attractive than a local road

# as a later step, I will rerun all these .py files with the updated weights. then I can recalculate routes and see whether
# the free flow travel times remain consistent

# the reason why I am assuming routes to be constant is that assuming varying routes would increase runtime and complexity
# I want to be able to iterate seamlessly on the weights

# -----------------

# Pulling functions from road_link_v3.py to update Free Flow Speed & Travel Time
# I want to directly change vehicles_df; I don't want to rerun calculate_initial_routes()

# Increasing a penalty increases speed and decreases duration
# I know this is counterintuitive
# it can be read as keeping x% of the original speed

# penalty_dictionary = {
#    'Minor-Minor': 0.7,
#    'Major-Major': 1.1,
#    'Minor-Link': 0.7,
#    'Link-Major': 0.6,
#    'Link-Link': 0.7,
#    'Link-Minor': 0.7,
#    'Minor-Major': 0.7,
#    'Minor-no_connections': 0.7,
#    'Link-no_connections': 0.7,
#    'Major-no_connections': 0.8,
#    'Major-Link': 0.7,
#    'Major-Minor': 0.7
# }

def create_penalty_weights(edges_df, penalty_dictionary):
    edges_df['penalty_weight'] = edges_df['category_matrix'].map(penalty_dictionary)
    return edges_df

# Free Flow Speed
# Units: Miles per Hour
def calculate_FFS(edges_df):
    FFS = edges_df['maxspeed'] * edges_df['penalty_weight']
    return FFS

# Unit: minutes
def calculate_free_flow_travel_time(edges_df):
    # FFS is in miles per hour so I need to convert it to miles per minute
    # length is in meters so I need to convert it to miles
    edges_df['length_miles'] = edges_df['length'] / 1609.34
    edges_df['free_flow_travel_time'] = (edges_df['length_miles']) / (edges_df['FFS'] / 60)
    return edges_df

def adjust_weights_for_edges_df(edges_df, penalty_dictionary):
    edges_df = create_penalty_weights(edges_df, penalty_dictionary)
    edges_df['FFS'] = calculate_FFS(edges_df)
    edges_df = calculate_free_flow_travel_time(edges_df)
    return edges_df

def main_recalculate_estimated_total_time_for_vehicles_df(vehicles_df, penalty_dictionary):
    # (1) Adjust the FFS and free_flow_travel_time for the saved edges_df based on the penalty dictionary
    edges_path = read_edges_path()
    edges_df = read_edges(edges_path)
    edges_df = adjust_weights_for_edges_df(edges_df, penalty_dictionary)
    # (2) Recalculate the estimated_total_time
    vehicles_df = add_estimated_total_time(vehicles_df, edges_df)
    return vehicles_df

# -----------------
# (6) Basic comparison of weights
# -----------------

# read the gmaps and vehicles_df data and compare

def read_gmaps_df():
    read_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'gmaps_free_flow_route_times.csv')
    gmaps_df = pd.read_csv(read_path)
    gmaps_df = gmaps_df.rename(columns={'duration_seconds': 'gmaps_duration_seconds'})
    return gmaps_df

def read_vehicles_df():
    read_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'vehicles.csv')
    vehicles_df = pd.read_csv(read_path)
    return vehicles_df

def merge_gmaps_and_vehicles(gmaps_df, vehicles_df):
    # gmaps_df = gmaps_df.rename(columns={'duration_seconds': 'gmaps_duration_seconds'})
    vehicles_df['estimated_duration_seconds'] = vehicles_df['estimated_total_time'] * 60 # convert minutes to seconds
    vehicles_df = vehicles_df.drop(columns='estimated_total_time')

    gmaps_df = gmaps_df[['person_id', 'gmaps_duration_seconds', 'origin_latitude', 'origin_longitude', 'destination_latitude', 'destination_longitude']].copy()
    vehicles_df = vehicles_df.merge(gmaps_df, 
                                    on=['person_id'], how='inner')
    return vehicles_df

def estimated_minus_gmaps(vehicles_df):
    vehicles_df['difference'] = vehicles_df['estimated_duration_seconds'] - vehicles_df['gmaps_duration_seconds']
    return vehicles_df

# -----------------
# (7) Setting weights
# -----------------

# Write a function that creates a list of penalty dictionaries based on multiple combinations of penalty weights 
# (for example, cross join Minor-minor 0.5 - 0.7 with a step of 0.1 with the rest of the weights)
# I like this approach because it gives me a lot of control in setting the weights. This is a simulation and I want it to make sense in the real world

# Then I can create a function that assesses the error rates (perhaps measure in % routes covered within 10min, 5min, 3min

# Rules
# No_connections should be treated the same
# Symmetric relationships should be treated the same -- except for links
# 	- Major-minor should be the same penalty as minor-major
# 	- The reason why links are excluded from this judgement is that sometimes you are increasing speed on a link and other times you are decreasing speed
# Major-major should be above 1. it should be the highest number
# Minor-minor should be below 1. it should be a low number. Perhaps not necessarily the lowest

# -----------------
# (7a) penalty combinations

# penalty_combination_dictionary = {
#    'Minor-Minor': [0.5, 0.8],
#    'Major-Major': [1, 1.3],
#    'Minor-Link': [0.8],
#    'Link-Major': [0.8],
#    'Link-Link': [0.8],
#    'Link-Minor': [0.8],
#    'Minor-Major': [0.5, 0.8],
#    'Minor-no_connections': [0.5, 0.8],
#    'Link-no_connections': [0.8],
#    'Major-no_connections': [0.8],
#    'Major-Link': [0.8],
#    'Major-Minor': [0.8]
# }

def generate_sequence(value_list, step=0.1):
    #1. generate sequence of numbers between min and max values with given step
    #2. if only one value in list, return that value
    #3. returns list of sequence numbers
    if len(value_list) == 1:
        return value_list
    min_val, max_val = value_list
    #4. using numpy arange for more precise float handling
    # adding step/2 ensures that the max value is included in the sequence
    return list(np.arange(min_val, max_val + step/2, step))

def generate_all_sequences(penalty_dict):
    #1. generate sequences for each key in the penalty dictionary
    #2. returns dictionary with all sequences
    return {key: generate_sequence(values) for key, values in penalty_dict.items()}

def generate_all_combinations(sequences_dict):
    #1. generate all possible combinations of values across all keys
    #2. returns pandas dataframe with all combinations
    keys = list(sequences_dict.keys())
    sequences = [sequences_dict[key] for key in keys]
    #3. generate all combinations using itertools.product
    combinations = list(product(*sequences))
    #4. convert to dataframe
    df = pd.DataFrame(combinations, columns=keys)
    return df

def create_penalty_combination_df(penalty_dict):
    #1. main function to generate all penalty combinations
    #2. returns pandas dataframe with all combinations
    sequences = generate_all_sequences(penalty_dict)
    result_df = generate_all_combinations(sequences)
    return result_df

# -----------------
# (7b) measuring success

# this code actually works, but it does take a while to run
# def iterate_and_measure_penalty_success(vehicles_df, gmaps_df, penalty_combination_df, minutes_list=[10, 5, 3]):
#     #1. loop through each penalty combination and calculate success metrics
#     #2. returns updated penalty_combination_df with success metrics for each time threshold
#     for time_minutes in minutes_list:
#         #3. create column name for this time threshold
#         col_name = f'num_within_{time_minutes}_min'
#         penalty_combination_df[col_name] = 0
    
#     #4. iterate through each row in penalty_combination_df
#     for idx, row in penalty_combination_df.iterrows():
#         #5. convert row to penalty dictionary
#         penalty_dict = row.to_dict()
        
#         #6. make a copy of vehicles_df to avoid modifying original
#         temp_vehicles_df = vehicles_df.copy()
        
#         #7. apply existing functions
#         temp_vehicles_df = main_recalculate_estimated_total_time_for_vehicles_df(
#             temp_vehicles_df, 
#             penalty_dict
#         )
#         temp_vehicles_df = merge_gmaps_and_vehicles(gmaps_df, temp_vehicles_df)
#         temp_vehicles_df = estimated_minus_gmaps(temp_vehicles_df)
        
#         #8. calculate success metrics for each time threshold
#         for time_minutes in minutes_list:
#             #9. convert minutes to seconds for comparison
#             time_seconds = time_minutes * 60
            
#             #10. count rows within threshold
#             within_threshold = (abs(temp_vehicles_df['difference']) <= time_seconds).sum()
            
#             #11. store result in penalty_combination_df
#             penalty_combination_df.at[idx, f'num_within_{time_minutes}_min'] = within_threshold
    
#     return penalty_combination_df

# this function is supposed to be faster
# def iterate_and_measure_penalty_success(vehicles_df, gmaps_df, penalty_combination_df, minutes_list=[10, 5, 3]):
#     # Pre-calculate seconds thresholds
#     seconds_thresholds = {min: min * 60 for min in minutes_list}
    
#     # Create all result columns at once
#     for time_minutes in minutes_list:
#         penalty_combination_df[f'num_within_{time_minutes}_min'] = 0
    
#     # Vectorize the operations by applying functions to groups
#     def process_penalty_combination(row):
#         # Convert row to dictionary once
#         penalty_dict = row.to_dict()
        
#         # Process vehicles data without creating full copy
#         # Step 1: Recalculate estimated times with new penalties
#         temp_vehicles_df = main_recalculate_estimated_total_time_for_vehicles_df(
#             vehicles_df,
#             penalty_dict
#         )
        
#         # Step 2: Merge with gmaps data and convert units
#         processed_df = merge_gmaps_and_vehicles(gmaps_df, temp_vehicles_df)
        
#         # Step 3: Calculate differences
#         processed_df = estimated_minus_gmaps(processed_df)
        
#         # Calculate results for all thresholds at once
#         results = {
#             f'num_within_{min}_min': (abs(processed_df['difference']) <= seconds_thresholds[min]).sum()
#             for min in minutes_list
#         }
        
#         return pd.Series(results)
    
#     # Apply the processing function to all rows at once
#     results = penalty_combination_df.apply(process_penalty_combination, axis=1)
    
#     # Update the original DataFrame with results
#     for col in results.columns:
#         penalty_combination_df[col] = results[col]
    
#     return penalty_combination_df

def process_single_combination(args):
    """Helper function to process a single penalty combination"""
    row, vehicles_df, gmaps_df, minutes_list = args
    
    # Convert series to dictionary
    penalty_dict = row.to_dict()
    seconds_thresholds = {min: min * 60 for min in minutes_list}
    
    # Process vehicles data
    temp_vehicles_df = main_recalculate_estimated_total_time_for_vehicles_df(
        vehicles_df,
        penalty_dict
    )
    processed_df = merge_gmaps_and_vehicles(gmaps_df, temp_vehicles_df)
    processed_df = estimated_minus_gmaps(processed_df)
    
    # Calculate results for all thresholds
    results = {
        f'num_within_{min}_min': (abs(processed_df['difference']) <= seconds_thresholds[min]).sum()
        for min in minutes_list
    }
    
    # Add index to identify the combination
    results['combination_idx'] = row.name
    return results

def iterate_and_measure_penalty_success(vehicles_df, gmaps_df, penalty_combination_df, minutes_list=[10, 5, 3], n_workers=None):
    """Parallelized version of penalty success measurement"""
    # If n_workers not specified, use number of CPU cores - 1 (leave one for system)
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)
    
    # Create arguments list for parallel processing
    args_list = [(row, vehicles_df, gmaps_df, minutes_list) 
                 for _, row in penalty_combination_df.iterrows()]
    
    # Process combinations in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_single_combination, args_list))
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('combination_idx')
    
    # Update original DataFrame with results
    for col in [f'num_within_{min}_min' for min in minutes_list]:
        penalty_combination_df[col] = results_df[col]
    
    return penalty_combination_df


# setting weights next step
# create an extension of edges_df
# AFTER vehicles_df, failures_df = calculate_initial_routes(G, od_matrix_sample, edges_df) 
# I can then pull edges_df['link_id', 'edge_category'] 
# I can also pull vehicles_df['route_links']
# then I can add columns to vehicles_df for the counts of each edge_category 
# NOTE: the routes will look different after I update the weights (for example, they can make highways more or less attractive)
