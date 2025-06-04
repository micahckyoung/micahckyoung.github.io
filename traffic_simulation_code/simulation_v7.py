# the purpose of this code is to fully simulate morning rush hour traffic on Oahu
# v2 includes debugging for 'route_links' which used to be both a string and a list
# I want route_links to always be a string representation of a list (e.g., '[1, 2, 3]', not an actual Python list)

# v3 includes a while loop if end_time isn't specified
# v3_test is a later version
# v4 adds more debugging statements
# v5 has additional debugging
# v6 updates the congestion speed equation
# v7 allows me to input my own od matrix

# import necessary packages
import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gpd
import datetime
import math
from typing import Dict, List, Tuple
import os
# import geopandas as gpd
import re
from datetime import datetime, timedelta
import json

# there will be the following datasets:

# (1) edges_df
# link_id, Node1, Node2, FFS, free_flow_travel_time, length, length_miles, capacity, lanes, following_distance, congestion_speed, congestion_tt, edge_category

# (2) time_intervals_df
# one column for each time interval (sorted ascending)
# example: 09:05:10 which is 9:05am and 10 seconds -- 'current_time'
# I'll write a function that will create time_intervals_df

# (3) in_queue_veh_df
# this is a subset of od_matrix. Its size will decrease over time until the df is empty
# person_id, origin_node, destination_node, departure_time

# (4) in_route_veh_df
# this marks the vehicles that are in route to their destination
# person_id, route_links, current_link, remaining_time_on_current_link, interval_length, fully_adjusted_in_step_a
# it should always be the case that current_link == route_links[0]
# interval_length is related to time_intervals_df (it is the distance between, so it could be 5 seconds)
# interval_length can change. For cars on existing links, interval_length can be subtracted from the time remaining on link
# there's movement for in_route before calculating congestion for the next interval
# if fully_adjusted_in_step_a is True, then remaining_time_on_current_link won't be updated
# - that's because remaining_time_on_current_link should refer to new intervals

# (5) completed_veh_df
# person_id, completion_time

# in the beginning I'm going to make a few simplifying assumptions:
# NOTE: these assumptions are helpful for reducing runtime
# - only run congestion function on primary roads (in the end, primary road congestion is all that I care about)
# -- this means that local roads will functionally have zero congestion (this could be a terrible assumption!)
# - measure departure in 5 min intervals
# -- with this assumption, there are 63 vehicles leaving orign node 110590771 all at once
# -- this assumption would be crazy (causing congestion immediatly on the route) if it weren't for the other assumption

# -----------------
# (1) Create the initial dataframes
# -----------------


# -----------------
# (1a) Read the original datasets

def read_od_matrix_path():
    read_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'od_matrix.csv')
    return read_path

def read_od_matrix(od_matrix_path):
    od_matrix = pd.read_csv(od_matrix_path)
    od_matrix['origin_node'] = od_matrix['origin_node'].astype(str)
    od_matrix['destination_node'] = od_matrix['destination_node'].astype(str)
    return od_matrix

def read_edges_path():
    edges_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'edges.gpkg')
    return edges_path

def read_edges(edges_path):
    edges = gpd.read_file(edges_path)
    return edges

# -----------------
# (1b) Adapt the original datasets

# capacity depends on road length, car length, following distance (speed & time gap), and number of lanes

# following distance measured in meters
# length measured in meters
# def create_following_distance(FFS):
#     # safety_time_gap is 2 seconds (2 second rule)
#     safety_time_gap = 2
#     # free flow speed is in miles per hour
#     # convert free flow speed to meters per second
#     # 1 mile = 1609.344 meters
#     # 1 hour = 3600 seconds
#     meters_per_second = (FFS * 1609.344) / 3600
#     following_distance = meters_per_second * safety_time_gap
#     return following_distance

def create_following_distance(FFS):
    # Convert FFS to m/s
    meters_per_second = (FFS * 1609.344) / 3600
    
    # Non-linear time gap calculation
    max_time_gap = 2.0
    min_time_gap = 1.0
    decay_factor = 0.02
    time_gap = min_time_gap + (max_time_gap - min_time_gap) * math.exp(-decay_factor * FFS)
    
    # Calculate following distance
    following_distance = meters_per_second * time_gap
    
    return following_distance

# capacity is measured in number of vehicles (rounding down)
def create_capacity(road_length, lanes, following_distance, avg_car_length = 4.5):
    # average car length: 4.5 meters
    capacity = (road_length / (avg_car_length + following_distance)) * lanes
    # NOTE: I might not want to round down
    return max(1, int(capacity))  # Using int() to round down. Ensure minimum capacity is 1

def create_congestion_speed(FFS, num_vehicles, original_capacity, road_length, lanes, weight, min_speed):
    # weight controls the steepness of the curve
    # when num_vehicles = capacity, congestion_speed = FFS
    if num_vehicles <= original_capacity + 0.0001:
        congestion_speed = FFS
    else:
        max_capacity = create_capacity(road_length, lanes, following_distance=1, avg_car_length = 4.5)
        # comparing fixed following distance capacity to originally calculated capacity
        # this is the x-intercept of the curve
        # y-axis is congestion speed, x-axis is % over capacity
        # subtract 1. 2/1 == 100% over capacity
        max_over_capacity = (max_capacity / original_capacity) - 1
        # if max_capacity is too close to original_capacity, I set a minimum value
        if max_over_capacity < 0.001:  # Small threshold to avoid division by near-zero
            max_over_capacity = 0.001
        # comparing vehicles on the road to capacity adjusting 1 so that it is measuring % over the 100% capacity
        over_capacity_pct = (num_vehicles / original_capacity) - 1
        # calculate what will be inside the natural log
        congestion_speed = FFS * (1 - ((over_capacity_pct / max_over_capacity) ** weight))
        if congestion_speed < min_speed:
            congestion_speed = min_speed
    return congestion_speed

# free_flow_travel_time is measured in minutes currently. I need it to be in seconds to match congestion travel time
def adjust_free_flow_travel_time_to_seconds(free_flow_travel_time):
    return free_flow_travel_time * 60

# free_flow_travel_time is measured in minutes currently
# I think congestion time should be measured in seconds because the road segments are really short
def create_congestion_time(congestion_speed, length_miles):
    # convert congestion_speed to miles per second
    # 60 seconds in a minute x 60 minutes in an hour
    congestion_speed_sec = congestion_speed / 3600
    congestion_time = length_miles / congestion_speed_sec
    return congestion_time

def create_time_interval_df(start_time='05:00:00', end_time='10:00:00', interval_length='5s'):
    # Create time intervals
    time_intervals = pd.date_range(start_time, end_time, freq=interval_length)
    # Create DataFrame and extract only time component
    time_intervals_df = pd.DataFrame(time_intervals.time, columns=['time_interval'])
    return time_intervals_df

# -----------------
# (1c) Create the initial dataframes (in_queue_veh_df, edges_df)

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

def initial_dfs_creation(od_matrix):
    # (1) read the dfs
    # od_matrix_path = read_od_matrix_path()
    # od_matrix = read_od_matrix(od_matrix_path)
    edges_path = read_edges_path()
    edges_df = read_edges(edges_path)

    # (2) create new columns for edges_df
    edges_df['free_flow_travel_time'] = edges_df.apply(lambda x: adjust_free_flow_travel_time_to_seconds(x['free_flow_travel_time']), axis=1)
    edges_df['following_distance'] = edges_df.apply(lambda x: create_following_distance(x['FFS']), axis=1)
    edges_df['capacity'] = edges_df.apply(lambda x: create_capacity(x['length'], x['lanes'], x['following_distance']), axis=1)
    edges_df['congestion_speed'] = edges_df['FFS'] # for now, congestion speed is the same as free flow speed
    edges_df['congestion_tt'] = edges_df.apply(lambda x: create_congestion_time(x['congestion_speed'], x['length_miles']), axis=1)

    # (3) prepare od_matrix and create in_queue_veh_df
    od_matrix = prepare_departure_time_for_od_matrix(od_matrix)
    queue_columns = ['person_id', 'origin_node', 'destination_node', 'departure_time']
    in_queue_veh_df = od_matrix[queue_columns].copy()

    edges_columns = ['link_id', 'Node1', 'Node2', 'FFS', 
                    'free_flow_travel_time', 'length', 'capacity', 'lanes', 'length_miles', 
                    'following_distance', 'congestion_speed', 'congestion_tt', 'edge_category'
                    ]
    edges_df = edges_df[edges_columns].copy()

    # (4) make sure that columns are saved with the correct dtype
    in_queue_veh_df['person_id'] = in_queue_veh_df['person_id'].astype(str)
    edges_df['link_id'] = edges_df['link_id'].astype(str)

    return in_queue_veh_df, edges_df

# -----------------
# (2) Steps on Loop
# -----------------

# Step A: Update the vehicles that are in route -- make sure that they make it through to the next link before getting affected by congestion
# Step B: Add new vehicles to in-route. These new vehicles will affect congestion
# Step C: Calculate/Update congestion for this time interval
# Step D: Update the vehicles that are in route -- clear them out completely (both newly joined and lingering)
# -- make sure that once route_links contains only 1 element, the vehicle is moved to completed_veh_df

# interval length is the length of the interval: ex. 5 seconds
# interval time is the time of the interval: ex. 09:05:10

# (2a) Update vehicles that are in route from the previous period
# -----------------

# the reason why I clear the route first is that future congestion won't affect the cars in the front of the line
# I don't want a situation like Zeno's Paradox where the car never reaches the end of the road segment because the 
# speed is always decreasing due to new cars piled up behind it

# pandas dataframes aren't good at storing complex objects like lists inside of their cells
# this has been the cause of some trouble, but all of my operations are based on dataframes, so I don't want to convert to a json format
# I'm going to use serialization and deserialization for in_route_veh_df['route_links']

# only run this if in_route_veh_df is not None
def clear_time_on_existing_link(in_route_veh_df):
    # Create a copy to avoid modifying the original DataFrame
    df = in_route_veh_df.copy()

    # Clean interval_length by removing 's' and converting to float
    df['interval_length'] = df['interval_length'].str.rstrip('s').astype(float)
    
    # Apply the logic row by row using loc
    # For vehicles that stay on current link
    # if remaining time is greater than the interval length, the current_link will stay the same
    # but the remaining time will decrease by the interval length
    mask = df['interval_length'] < df['remaining_time_on_current_link']
    df.loc[mask, 'remaining_time_on_current_link'] -= df.loc[mask, 'interval_length']
    df.loc[mask, 'fully_adjusted_in_step_a'] = True # I will need to make sure to exclude these rows from the future functions
    df.loc[mask, 'interval_length'] = 0 # there's no more time left in the period for these cars
    
    # For vehicles that move to next link
    # if remaining time is less than the interval length, the vehicle will move to the next link
    # interval_length will decrease by that remaining time

    mask_move = df['interval_length'] >= df['remaining_time_on_current_link']

    # First handle vehicles moving to next link (those with more links remaining)
    has_next_link = df['route_links'].apply(lambda x: len(get_route_links(x)) > 1)
    # Use get_route_links to access next link
    df.loc[mask_move & has_next_link, 'current_link'] = df.loc[mask_move & has_next_link, 'route_links'].apply(
        lambda x: get_route_links(x)[1]
    )
    # Use get_route_links when slicing routes and re-serializing
    df.loc[mask_move & has_next_link, 'route_links'] = df.loc[mask_move & has_next_link, 'route_links'].apply(
        lambda x: json.dumps(get_route_links(x)[1:])
    )

    # Then handle vehicles completing their last link
    completing = mask_move & ~has_next_link
    df.loc[completing, 'current_link'] = 'complete'
    # Use json.dumps with empty list for completed routes
    df.loc[completing, 'route_links'] = df.loc[completing, 'route_links'].apply(
        lambda x: json.dumps([])
    )
    # Update interval_length (this stays the same for both cases)
    df.loc[mask_move, 'interval_length'] -= df.loc[mask_move, 'remaining_time_on_current_link']

    return df

# (2b) Add new vehicles to in-route
# -----------------

# I need to build a helper function that will convert the route_links string to a list
# added some error logging to help with debugging
# I use json.dumps() to convert the route_link list to a JSON string. Then I need to use json.loads() to convert it back to a list

def get_route_links(route_links_str):
    """Convert route_links string to a list, with error logging"""
    if pd.isna(route_links_str):
        return []
    
    try:
        return json.loads(route_links_str)
    except json.JSONDecodeError as e:
        # Log specific JSON error and try fallback
        print(f"Warning: JSON parsing failed with error: {e}. Trying fallback method.")
        
        try:
            # Only use eval as fallback for transition period
            result = eval(route_links_str)
            if isinstance(result, list):
                return result
            else:
                print(f"Warning: eval did not return a list. Got {type(result)} instead.")
                return []
        except Exception as e:
            # Log specific eval error
            print(f"Warning: Could not parse route_links: {route_links_str[:50]}... Error: {e}")
            return []

# the below code assigns new vehicles to routes
# this needs to be rerun because congestion_tt changes over time

def create_network_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
    #(1) Create directed graph from edges dataframe using free_flow_travel_time as weight
    G = nx.DiGraph()
    #(2) Add edges with their attributes
    for _, row in edges_df.iterrows():
        G.add_edge(
            str(row['Node1']),
            str(row['Node2']),
            weight=row['congestion_tt'],
            length=row['length'], # in meters
            length_miles=row['length_miles'],
            FFS=row['FFS'],
            capacity=row['capacity'],
            edge_category=row['edge_category'],
            link_id=row['link_id']
        )
    #(3) Basic validation
    # print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

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
    # print(f"Calculating paths for {len(unique_od_pairs)} unique OD pairs...")
    
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
    # success_count = (vehicles_df['routing_status'] == 'success').sum()
    # fail_count = (vehicles_df['routing_status'] != 'success').sum()
    # print(f"\nPath Calculation Summary:")
    # print(f"Successful paths: {success_count}")
    # print(f"Failed paths: {fail_count}")
    
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

    # NEW
    for idx, row in vehicles_df[mask].iterrows():
        try:
            node_path = eval(row['node_path'])
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
                # Ensure proper string formatting with repr()
                # vehicles_df.loc[idx, 'route_links'] = repr(path_links)  # Changed from str() to repr()
                vehicles_df.loc[idx, 'route_links'] = json.dumps(path_links)
                vehicles_df.loc[idx, 'conversion_status'] = 'success'
                success_count += 1
            else:
                vehicles_df.loc[idx, 'conversion_status'] = 'discontinuous'
                fail_count += 1
                
        except Exception as e:
            vehicles_df.loc[idx, 'conversion_status'] = f'error: {str(e)}'
            fail_count += 1
    
    return vehicles_df

# # Fix for the concatenation warning
# def safe_concat(df1, df2):
#     if df1.empty:
#         return df2.copy()
#     if df2.empty:
#         return df1.copy()
    
#     # Ensure consistent column presence and dtypes
#     for col in df1.columns:
#         if col not in df2.columns:
#             df2[col] = pd.NA  # Add missing column with NA values
    
#     for col in df2.columns:
#         if col not in df1.columns:
#             df1[col] = pd.NA  # Add missing column with NA values
    
#     # Make sure both DataFrames have the same column order
#     df2 = df2[df1.columns]
    
#     # Now concat them
#     return pd.concat([df1, df2], ignore_index=True)

# def safe_concat(df1, df2):
#     # Handle empty dataframes
#     if df1.empty:
#         return df2.copy()
#     if df2.empty:
#         return df1.copy()
    
#     # Create a copy of df2 to avoid the SettingWithCopyWarning
#     df2_copy = df2.copy()
    
#     # Ensure consistent column presence and dtypes
#     for col in df1.columns:
#         if col not in df2_copy.columns:
#             # Use .loc to properly set values
#             df2_copy.loc[:, col] = pd.NA  # Add missing column with NA values
    
#     # Create a copy of df1 to avoid the SettingWithCopyWarning
#     df1_copy = df1.copy()
    
#     for col in df2_copy.columns:
#         if col not in df1_copy.columns:
#             # Use .loc to properly set values
#             df1_copy.loc[:, col] = pd.NA  # Add missing column with NA values
    
#     # Make sure both DataFrames have the same column order
#     df2_copy = df2_copy[df1_copy.columns]
    
#     # Now concat them
#     return pd.concat([df1_copy, df2_copy], ignore_index=True)

def safe_concat(df1, df2):
    if df1.empty:
        return df2.copy()
    if df2.empty:
        return df1.copy()
    
    # Convert to records (list of dictionaries)
    records1 = df1.to_dict('records')
    records2 = df2.to_dict('records')
    
    # Combine records
    all_records = records1 + records2
    
    # Create new DataFrame
    return pd.DataFrame(all_records)

# only run this where the time_intervals_df == the minute marker
# only run this when current_time matches the dispatch interval
def add_new_vehicles_to_in_route(in_queue_veh_df, current_time, edges_df, G, in_route_veh_df, interval_length):
    # (1) determine the relevant queue vehicles 
    current_time_str = current_time.strftime('%H:%M:%S')
    current_veh = in_queue_veh_df[in_queue_veh_df['departure_time'].astype(str) == current_time_str].copy()
    in_queue_veh_df = in_queue_veh_df[in_queue_veh_df['departure_time'].astype(str) != current_time_str].copy()
    # (2) update G network graph with new congestion_tt from edges_df

    # I'm going to use the already created network graph and merely adjust the weight (hopefully this saves time in the future)
    # G = create_network_graph(edges_df)

    new_weights = {(row['Node1'], row['Node2']): row['congestion_tt'] 
                   for _, row in edges_df.iterrows()}
    nx.set_edge_attributes(G, new_weights, 'weight')

    # (3) calculate optimal routes for these vehicles (i.e. create current_veh['route_links'])
    current_veh = calculate_node_paths(G, current_veh)
    current_veh = convert_node_paths_to_links(current_veh, edges_df)

    current_veh['current_link'] = current_veh['route_links'].apply(lambda x: get_route_links(x)[0] if get_route_links(x) else None)

    current_veh['interval_length'] = interval_length
    current_veh['fully_adjusted_in_step_a'] = False

    # (4) concatenate back to in_route_veh_df
    columns_to_include = ['person_id', 'route_links', 'current_link', 'interval_length', 'fully_adjusted_in_step_a']
    # in_route_veh_df = pd.concat([in_route_veh_df, current_veh[columns_to_include]], axis=0)
    in_route_veh_df = safe_concat(in_route_veh_df, current_veh[columns_to_include])

    return in_route_veh_df, in_queue_veh_df

# (2c) Update Congestion
# -----------------

def update_congestion_in_edges_df(edges_df, in_route_veh_df, weight, min_speed, for_all_road_types):

    # (1) Filter out anything that doesn't affect congestion
    
    # (1a) Filter out completed vehicles [Step A creates completed vehicles]
    in_route_veh_df_not_complete = in_route_veh_df[in_route_veh_df['current_link'] != 'complete'].copy()
    if in_route_veh_df_not_complete.empty:
        # if all vehicles have completed their routes, there are no cars on the road = no congestion
        edges_df['congestion_tt'] = edges_df['free_flow_travel_time']
        edges_df['congestion_speed'] = edges_df['FFS']
        return edges_df
        
    # (1b) filter for edges with vehicles currently on them
    current_edges = edges_df[edges_df['link_id'].isin(in_route_veh_df_not_complete['current_link'])].copy()
    if current_edges.empty:
        # if there are no cars on the road = no congestion
        edges_df['congestion_tt'] = edges_df['free_flow_travel_time']
        edges_df['congestion_speed'] = edges_df['FFS']
        return edges_df
    
    # (1c) filter for edges where number of vehicles is greater than capacity
    # Create vehicle counts
    num_vehicles_df = in_route_veh_df_not_complete.groupby('current_link', as_index=False).size()
    num_vehicles_df.columns = ['link_id', 'num_vehicles']
    # Merge with explicit column names
    current_edges = current_edges.merge(
        num_vehicles_df[['link_id', 'num_vehicles']], 
        on='link_id', 
        how='left',
        validate='1:1'  # Ensure one-to-one merge
    )
    # see which edges have more vehicles than capacity
    current_edges = current_edges[current_edges['num_vehicles'] > current_edges['capacity'] + 0.0001].copy()
    if current_edges.empty:
        # if there are no links above capacity, there is no congestion
        edges_df['congestion_tt'] = edges_df['free_flow_travel_time']
        edges_df['congestion_speed'] = edges_df['FFS']
        return edges_df
    
    # (1d) filter for Major roads (based on for_all_road_types)
    if not for_all_road_types:
        current_edges = current_edges[current_edges['edge_category'] == 'Major'].copy()
    if current_edges.empty:
        # if there are no links above capacity, there is no congestion
        edges_df['congestion_tt'] = edges_df['free_flow_travel_time']
        edges_df['congestion_speed'] = edges_df['FFS']
        return edges_df
    
    # (2) Update congestion characteristics for the links that are left
    current_edges['congestion_speed'] = current_edges.apply(
        lambda x: create_congestion_speed(x['FFS'], x['num_vehicles'], x['capacity'], x['length'], x['lanes'], weight, min_speed), 
        axis=1
    )
    current_edges['congestion_tt'] = current_edges.apply(
        lambda x: create_congestion_time(x['congestion_speed'], x['length_miles']), 
        axis=1
    )
    # drop number of vehicles
    current_edges = current_edges.drop('num_vehicles', axis=1)
    edges_df_removed_congestion = edges_df[~edges_df['link_id'].isin(current_edges['link_id'])].copy()
    # I need to reset links that aren't congested back to their free flow travel time
    edges_df_removed_congestion['congestion_tt'] = edges_df_removed_congestion['free_flow_travel_time']
    edges_df_removed_congestion['congestion_speed'] = edges_df_removed_congestion['FFS']
    # edges_df_new = pd.concat([edges_df_removed_congestion, current_edges], ignore_index=True)
    edges_df_new = safe_concat(edges_df_removed_congestion, current_edges)
    
    return edges_df_new

# (2d) Assign in route vehicles to their next edges
# -----------------

def process_single_route(congestion_lookup, route_links_list, interval_length):
    # (1) if there are no links in the route, handle it
    if not route_links_list:  
        return [], 0
        
    # (2) Convert interval_length from '60s' format to float
    if isinstance(interval_length, str):
        interval_length = float(interval_length.replace('s', ''))
            
    # (3) Convert route links to congestion times
    try:
        link_times = np.array([congestion_lookup[link] for link in route_links_list])
        cumulative_times = np.cumsum(link_times)
    except KeyError as e:
        print(f"KeyError in process_single_route: {e}")
        print(f"route_links: {route_links_list}")
        print(f"congestion_lookup keys sample: {list(congestion_lookup.keys())[:5]}")
        return [], 0
        
    # # (3) Find the last link one can fully traverse
    # mask = cumulative_times <= interval_length
    # if all(mask):
    #     return [], cumulative_times[-1] - interval_length
    # if not any(mask):  # Can't complete even first link
    #     return route_links_list, interval_length

    # NEW: handle floating point numbers 
    # (3) Find the last link one can fully traverse
    mask = cumulative_times <= interval_length + 1e-6  # Add small epsilon
    if all(mask):
        return [], max(0, cumulative_times[-1] - interval_length)
    if not any(mask):  # Can't complete even first link
        return route_links_list, interval_length
        
    # current_link_idx is the index of first False (first link we can't complete)
    current_link_idx = np.where(~mask)[0][0]
    # (4) Calculate remaining time on next link (if any)
    remaining_time = cumulative_times[current_link_idx] - interval_length
    # (5) Update route_links_list (removing traversed links)
    remaining_links_list = route_links_list[current_link_idx:]
    return remaining_links_list, remaining_time

def process_vehicle_routes(in_route_veh_df, edges_df):
    """Process vehicle routes using congestion data from edges_df"""
    # Create a lookup dictionary for quick access to congestion times
    congestion_lookup = edges_df.set_index('link_id')['congestion_tt'].to_dict()
    
    # Initialize result columns
    new_routes = []
    remaining_times = []
    
    # Process each row
    for _, row in in_route_veh_df.iterrows():
        # Convert route_links string to list using your existing function
        route_links = get_route_links(row['route_links'])
        
        # Process the route - process_single_route already handles empty lists
        remaining_links, remaining_time = process_single_route(
            congestion_lookup, 
            route_links, 
            row['interval_length']
        )
        new_routes.append(remaining_links)
        remaining_times.append(remaining_time)
    
    # Update the dataframe with the new values
    in_route_veh_df['route_links'] = [json.dumps(route) for route in new_routes]
    in_route_veh_df['remaining_time_on_current_link'] = remaining_times
    
    return in_route_veh_df

def get_current_link(route_links_str):
    route_links = get_route_links(route_links_str)
    return 'complete' if len(route_links) == 0 else route_links[0]

# def assign_in_route_to_next_edges(in_route_veh_df_old, edges_df, interval_length, current_time, completed_veh_df):

#     in_route_veh_df = in_route_veh_df_old.copy()
#     # (1) only work with vehicles that haven't already been on the edge in the previous interval
#     # this should be the same as 'interval_length' != 0
#     # I'm also filtering for vehicles that haven't completed their route
#     in_route_filtered = in_route_veh_df[
#         (in_route_veh_df['fully_adjusted_in_step_a'] == False) & 
#         (in_route_veh_df['current_link'] != 'complete')
#     ].copy()
#     # in_route_filtered = in_route_veh_df[(in_route_veh_df['fully_adjusted_in_step_a'] == False) | (in_route_veh_df['current_link'] != 'complete')].copy()
#     # (2) process the routes
#     # now I have updated 'route_links', 'remaining_time_on_current_link'
#     in_route_filtered = process_vehicle_routes(in_route_filtered, edges_df)
#     in_route_filtered['current_link'] = in_route_filtered['route_links'].apply(get_current_link)
#     # (3) update in_route_veh_df (add the filtered back in)
#     in_route_veh_df = in_route_veh_df[in_route_veh_df['fully_adjusted_in_step_a'] == True].copy()
#     # in_route_veh_df = pd.concat([in_route_veh_df, in_route_filtered], ignore_index=True)
#     in_route_veh_df = safe_concat(in_route_veh_df, in_route_filtered)
#     # (4) reset parameters
#     in_route_veh_df['interval_length'] = interval_length
#     in_route_veh_df['fully_adjusted_in_step_a'] = False
#     # (5) create the completed_veh_df
#     add_completed = in_route_veh_df_old[in_route_veh_df_old['current_link'] == 'complete'].copy()
#     in_route_veh_df = in_route_veh_df[in_route_veh_df['current_link'] != 'complete'].copy()
#     add_completed['completion_time'] = current_time
#     # completed_veh_df = pd.concat([completed_veh_df, add_completed[['person_id', 'completion_time']]], axis=0)
#     completed_veh_df = safe_concat(completed_veh_df, add_completed[['person_id', 'completion_time']])

#     # Debug counts
#     total_rows = len(in_route_veh_df) + len(completed_veh_df)
#     print(f"Debug: in_route_veh_df rows: {len(in_route_veh_df)}, completed_veh_df rows: {len(completed_veh_df)}, total: {total_rows}")
#     print(f"Debug: completed this step: {len(add_completed)}")

#     return in_route_veh_df, completed_veh_df

def assign_in_route_to_next_edges(in_route_veh_df, edges_df, interval_length, current_time, completed_veh_df):
    # (1) Handle any vehicles that are already marked as complete
    already_complete = in_route_veh_df[in_route_veh_df['current_link'] == 'complete'].copy()
    if not already_complete.empty:
        already_complete['completion_time'] = current_time
        completed_veh_df = safe_concat(completed_veh_df, already_complete[['person_id', 'completion_time']])
        # Remove these from in_route_veh_df
        in_route_veh_df = in_route_veh_df[in_route_veh_df['current_link'] != 'complete'].copy()
    
    # (2) Only work with vehicles that haven't already been on the edge in the previous interval
    in_route_filtered = in_route_veh_df[in_route_veh_df['fully_adjusted_in_step_a'] == False].copy()
    
    # (3) Process the routes
    in_route_filtered = process_vehicle_routes(in_route_filtered, edges_df)
    in_route_filtered['current_link'] = in_route_filtered['route_links'].apply(get_current_link)
    
    # (4) Update in_route_veh_df (add the filtered back in)
    in_route_veh_df = in_route_veh_df[in_route_veh_df['fully_adjusted_in_step_a'] == True].copy()
    in_route_veh_df = safe_concat(in_route_veh_df, in_route_filtered)
    
    # (5) Reset parameters
    in_route_veh_df['interval_length'] = interval_length
    in_route_veh_df['fully_adjusted_in_step_a'] = False
    
    # (6) Now handle any newly completed vehicles
    newly_complete = in_route_veh_df[in_route_veh_df['current_link'] == 'complete'].copy()
    if not newly_complete.empty:
        newly_complete['completion_time'] = current_time
        completed_veh_df = safe_concat(completed_veh_df, newly_complete[['person_id', 'completion_time']])
        # Remove these from in_route_veh_df
        in_route_veh_df = in_route_veh_df[in_route_veh_df['current_link'] != 'complete'].copy()
    
    return in_route_veh_df, completed_veh_df

# def assign_in_route_to_next_edges(in_route_veh_df, edges_df, interval_length, current_time, completed_veh_df):
#     # Initial count tracking
#     initial_route_count = len(in_route_veh_df)
#     initial_completed_count = len(completed_veh_df)
    
#     # (1) Filter vehicles to process
#     in_route_filtered = in_route_veh_df[
#         (in_route_veh_df['fully_adjusted_in_step_a'] == False) & 
#         (in_route_veh_df['current_link'] != 'complete')
#     ].copy()
    
#     # Count vehicles not processed in this step
#     in_route_adjusted = in_route_veh_df[in_route_veh_df['fully_adjusted_in_step_a'] == True].copy()
#     in_route_already_complete = in_route_veh_df[in_route_veh_df['current_link'] == 'complete'].copy()
    
#     # Verify filtering preserved all rows
#     filter_sum = len(in_route_filtered) + len(in_route_adjusted) + len(in_route_already_complete)
#     if initial_route_count != filter_sum:
#         print(f"ERROR: Initial filtering lost rows! Initial: {initial_route_count}, After: {filter_sum}")
#         print(f"Filtered: {len(in_route_filtered)}, Adjusted: {len(in_route_adjusted)}, Already Complete: {len(in_route_already_complete)}")
    
#     # (2) Process routes for filtered vehicles
#     filtered_before_processing = len(in_route_filtered)
#     in_route_filtered = process_vehicle_routes(in_route_filtered, edges_df)
#     if len(in_route_filtered) != filtered_before_processing:
#         print(f"ERROR: process_vehicle_routes changed row count! Before: {filtered_before_processing}, After: {len(in_route_filtered)}")
    
#     # Update current link for filtered vehicles
#     in_route_filtered['current_link'] = in_route_filtered['route_links'].apply(get_current_link)
    
#     # (3) Recombine adjusted and filtered vehicles
#     before_concat = len(in_route_adjusted) + len(in_route_filtered)
#     in_route_veh_df = safe_concat(in_route_adjusted, in_route_filtered)
#     if len(in_route_veh_df) != before_concat:
#         print(f"ERROR: safe_concat lost rows! Before: {before_concat}, After: {len(in_route_veh_df)}")
    
#     # Check if already-complete vehicles were dropped
#     if len(in_route_already_complete) > 0:
#         print(f"WARNING: {len(in_route_already_complete)} already complete vehicles may have been dropped!")
    
#     # (4) Reset parameters
#     in_route_veh_df['interval_length'] = interval_length
#     in_route_veh_df['fully_adjusted_in_step_a'] = False
    
#     # (5) Move completed vehicles
#     # Count before completion processing
#     route_before_completion = len(in_route_veh_df)
    
#     # Extract all completed vehicles
#     add_completed = in_route_veh_df[in_route_veh_df['current_link'] == 'complete'].copy()
#     print(f"DEBUG: Found {len(add_completed)} newly completed vehicles")
    
#     # Remove completed from in-route
#     in_route_veh_df = in_route_veh_df[in_route_veh_df['current_link'] != 'complete'].copy()
    
#     # Verify completion extraction was clean
#     if route_before_completion != (len(in_route_veh_df) + len(add_completed)):
#         print(f"ERROR: Completion extraction lost rows! Before: {route_before_completion}, After: {len(in_route_veh_df) + len(add_completed)}")
    
#     # Add to completed with completion time
#     add_completed['completion_time'] = current_time
    
#     # Check person_id of completed before attempting to add
#     if len(add_completed) > 0:
#         print(f"DEBUG: First few person_ids being completed: {add_completed['person_id'].head(3).tolist()}")
#         print(f"DEBUG: Duplicate person_ids in completed batch: {add_completed['person_id'].duplicated().sum()}")
    
#     # Here's likely where the issue is happening - check column presence and types
#     print(f"DEBUG: Columns in add_completed: {add_completed.columns.tolist()}")
#     print(f"DEBUG: Columns in completed_veh_df: {completed_veh_df.columns.tolist()}")
    
#     # Try to prepare data more carefully
#     complete_data = add_completed[['person_id', 'completion_time']].copy()
    
#     # Check data before concat
#     completed_before = len(completed_veh_df)
#     completed_to_add = len(complete_data)
    
#     # Add to completed
#     completed_veh_df = safe_concat(completed_veh_df, complete_data)
    
#     # Check if adding to completed worked properly
#     if len(completed_veh_df) != (completed_before + completed_to_add):
#         print(f"ERROR: Adding to completed_veh_df lost rows! Before: {completed_before}, Added: {completed_to_add}, After: {len(completed_veh_df)}")
#         # Check for duplicate person_ids that might cause issues
#         if 'person_id' in completed_veh_df.columns:
#             dupes = completed_veh_df['person_id'].duplicated().sum()
#             print(f"DEBUG: Duplicated person_ids in final completed_veh_df: {dupes}")
    
#     # Final verification
#     final_count = len(in_route_veh_df) + len(completed_veh_df)
#     initial_count = initial_route_count + initial_completed_count
#     if final_count != initial_count:
#         print(f"ERROR: Final row count mismatch! Initial: {initial_count}, Final: {final_count}, Diff: {initial_count - final_count}")
#         # Look for lost rows by tracing every transition
#         print(f"Initial in_route: {initial_route_count}, Final in_route: {len(in_route_veh_df)}, Diff: {initial_route_count - len(in_route_veh_df)}")
#         print(f"Initial completed: {initial_completed_count}, Final completed: {len(completed_veh_df)}, Diff: {len(completed_veh_df) - initial_completed_count}")
#         print(f"Newly completed this step: {len(add_completed)}")
    
#     return in_route_veh_df, completed_veh_df

# -----------------
# (3) Main Loop Functions
# -----------------

def find_queue_dispatch_frequency(in_queue_veh_df):
    # Sort departure times in ascending order and drop any duplicates
    sorted_times = pd.to_datetime(
        in_queue_veh_df['departure_time'],
        format='%H:%M:%S').sort_values().drop_duplicates()
    # Calculate time differences between consecutive departures
    time_differences = sorted_times.diff()
    # Find the minimum difference, excluding NA values
    min_difference = time_differences.dropna().min()
    return min_difference

# checking that the queue dispatch time interval matches the current time
def is_on_interval(current_time, dispatch_interval_timedelta):
    # Convert current_time to timedelta since midnight for easy comparison
    if isinstance(current_time, str):
        time_obj = pd.to_datetime(current_time, format='%H:%M:%S')
    else:
        time_obj = pd.to_datetime(current_time.strftime('%H:%M:%S'), format='%H:%M:%S')
    
    time_delta_since_start = pd.Timedelta(hours=time_obj.hour, 
                                         minutes=time_obj.minute, 
                                         seconds=time_obj.second)
    
    # Check if current time is divisible by the interval with no remainder
    return time_delta_since_start % dispatch_interval_timedelta == pd.Timedelta(0)

def create_next_time_unit(current_time, interval_length):
    # Convert time to datetime string (need a dummy date)
    current_datetime_str = f'2000-01-01 {current_time}'
    
    # Create a range with just two points - current and next
    time_range = pd.date_range(current_datetime_str, periods=2, freq=interval_length)
    
    # Return just the time component of the second point
    return time_range[1].time()

# def run_full_loop(edges_df, in_queue_veh_df, weight=1, min_speed=5, for_all_road_types=False, start_time='05:00:00', end_time = None, interval_length='5s'):

#     # (1) preparation regardless of end_time
#     # (1a) create initial in_route_veh_df and completed_veh_df
#     in_route_veh_df = pd.DataFrame(columns=['person_id', 'route_links', 'current_link', 'remaining_time_on_current_link', 'interval_length', 'fully_adjusted_in_step_a'])
#     completed_veh_df = pd.DataFrame(columns=['person_id', 'completion_time'])
#     # (1b) create network graph
#     G = create_network_graph(edges_df)
#     # (1c) find the dispatch interval time (i.e. how frequently we are recording vehicles leaving their starting positions)
#     dispatch_interval_time = find_queue_dispatch_frequency(in_queue_veh_df)

#     # (2) preparation if end_time is given
#     if end_time is not None: 
#         print("end_time provided")
#         # (2a) create time_intervals_df
#         time_intervals_df = create_time_interval_df(start_time, end_time, interval_length)
        
#         # (2b) run the loop
#         for current_time in time_intervals_df['time_interval']:
#             # add a helpful print statement
#             if current_time.minute == 0 and current_time.second == 0:
#                 print(f"Hour reached: {current_time.strftime('%H:%M:%S')}")

#             # Step A: Update the vehicles that are in route
#             if not in_route_veh_df.empty: # only run this if in_route_veh_df isn't an empty dataframe
#                 # print('Step A')
#                 in_route_veh_df = clear_time_on_existing_link(in_route_veh_df)
#             # Step B: Add new vehicles to in-route
#             # only run this where the time_intervals_df == the minute marker
#             if is_on_interval(current_time, dispatch_interval_time):
#                 # print('Step B')
#                 in_route_veh_df, in_queue_veh_df = add_new_vehicles_to_in_route(in_queue_veh_df, current_time, edges_df, G, in_route_veh_df, interval_length)

#             # Step C: Calculate/Update congestion for this time interval
#             # print('Step C')
#             edges_df = update_congestion_in_edges_df(edges_df, in_route_veh_df, weight, min_speed, for_all_road_types)
#             # Step D: Update the vehicles that are in route
#             # print('Step D')
#             in_route_veh_df, completed_veh_df = assign_in_route_to_next_edges(in_route_veh_df, edges_df, interval_length, current_time, completed_veh_df)

#     # (3) preparation if end_time is not provided
#     else:
#         print("end_time not provided. Looping until both in_route_veh_df AND queue_veh_df are empty")

#         current_time = pd.to_datetime(start_time).time()

#         while not in_route_veh_df.empty or not in_queue_veh_df.empty:
#         # while not in_route_veh_df.empty and not in_queue_veh_df.empty:
#         # while (in_route_veh_df.shape[0] != 0) & (in_queue_veh_df.shape[0] != 0):

#             # add a helpful print statement
#             if current_time.minute == 0 and current_time.second == 0:
#                 print(f"Hour reached: {current_time.strftime('%H:%M:%S')}")

#             # Step A: Update the vehicles that are in route
#             if not in_route_veh_df.empty: # only run this if in_route_veh_df isn't an empty dataframe
#                 # print('Step A')
#                 in_route_veh_df = clear_time_on_existing_link(in_route_veh_df)
#             # Step B: Add new vehicles to in-route
#             # only run this where the time_intervals_df == the minute marker
#             if is_on_interval(current_time, dispatch_interval_time):
#                 # print('Step B')
#                 in_route_veh_df, in_queue_veh_df = add_new_vehicles_to_in_route(in_queue_veh_df, current_time, edges_df, G, in_route_veh_df, interval_length)

#             # Step C: Calculate/Update congestion for this time interval

#             edges_df = update_congestion_in_edges_df(edges_df, in_route_veh_df, weight, min_speed, for_all_road_types)

#             # print("edges_df after Step C:", edges_df.shape if edges_df is not None else "None")
#             # Step D: Update the vehicles that are in route
#             # print('Step D')
#             in_route_veh_df, completed_veh_df = assign_in_route_to_next_edges(in_route_veh_df, edges_df, interval_length, current_time, completed_veh_df)

#             # update current_time
#             current_time = create_next_time_unit(current_time, interval_length)
#             # completed_veh_df['test_completion_time'] = completed_veh_df['test_completion_time'].fillna(current_time)
#             # print(current_time)

#     return edges_df, in_route_veh_df, completed_veh_df

# write function to check that row counts match
# def match_row_counts(step_letter, original_number_of_rows, in_queue_veh_df, in_route_veh_df, completed_veh_df, current_time):
#     if original_number_of_rows != in_queue_veh_df.shape[0] + in_route_veh_df.shape[0] + completed_veh_df.shape[0]:
#         print(f"Current time: {current_time.strftime('%H:%M:%S')}, Step {step_letter}: Row counts do not match")
#     else:
#         print(f"Current time: {current_time.strftime('%H:%M:%S')}, Step {step_letter}: Row counts match")
#     return

def match_row_counts(step_letter, original_number_of_rows, in_queue_veh_df, in_route_veh_df, completed_veh_df, current_time):
    total_current_rows = in_queue_veh_df.shape[0] + in_route_veh_df.shape[0] + completed_veh_df.shape[0]
    
    if original_number_of_rows != total_current_rows:
        print(f"Current time: {current_time.strftime('%H:%M:%S')}, Step {step_letter}: Row counts do not match")
        print(f"Original: {original_number_of_rows}, Current total: {total_current_rows}")
        print(f"Queue: {in_queue_veh_df.shape[0]}, In-route: {in_route_veh_df.shape[0]}, Completed: {completed_veh_df.shape[0]}")
    # else:
    #     print(f"Current time: {current_time.strftime('%H:%M:%S')}, Step {step_letter}: Row counts match")
    return

def match_edge_counts(original_number_of_edges, edges_df, current_time):
    if original_number_of_edges != edges_df.shape[0]:
        print(f"Current time: {current_time.strftime('%H:%M:%S')}, Edge counts do not match")
    return

def process_single_time_step(current_time, in_route_veh_df, in_queue_veh_df, edges_df, G, 
                             dispatch_interval_time, interval_length, weight, min_speed, 
                             for_all_road_types, completed_veh_df, original_number_of_rows, original_number_of_edges):
    """
    Process a single time step in the traffic simulation.
    
    Args:
        current_time: The current time in the simulation
        in_route_veh_df: DataFrame of vehicles currently in route
        in_queue_veh_df: DataFrame of vehicles waiting to be dispatched
        edges_df: DataFrame of network edges
        G: Network graph
        dispatch_interval_time: How frequently vehicles are dispatched
        interval_length: Length of the time interval
        weight: Weight parameter for congestion calculation
        min_speed: Minimum speed parameter
        for_all_road_types: Boolean indicating whether to apply to all road types
        completed_veh_df: DataFrame of completed vehicle trips
        
    Returns:
        Tuple of (in_route_veh_df, in_queue_veh_df, edges_df, completed_veh_df)
    """
    # Add a helpful print statement for hour marks
    if current_time.minute == 0 and current_time.second == 0:
        print(f"Hour reached: {current_time.strftime('%H:%M:%S')}")

    # Step A: Update the vehicles that are in route
    if not in_route_veh_df.empty:  # only run this if in_route_veh_df isn't an empty dataframe
        in_route_veh_df = clear_time_on_existing_link(in_route_veh_df)
        # check that the row counts match
        match_row_counts('A', original_number_of_rows, in_queue_veh_df, in_route_veh_df, completed_veh_df, current_time)
    
    # Step B: Add new vehicles to in-route
    # only run this where the time_intervals_df == the minute marker
    if is_on_interval(current_time, dispatch_interval_time):
        in_route_veh_df, in_queue_veh_df = add_new_vehicles_to_in_route(
            in_queue_veh_df, current_time, edges_df, G, in_route_veh_df, interval_length
        )
        # check that the row counts match
        match_row_counts('B', original_number_of_rows, in_queue_veh_df, in_route_veh_df, completed_veh_df, current_time)

    # Step C: Calculate/Update congestion for this time interval
    edges_df = update_congestion_in_edges_df(
        edges_df, in_route_veh_df, weight, min_speed, for_all_road_types
    )
    # check that edges match
    match_edge_counts(original_number_of_edges, edges_df, current_time)
    
    # Step D: Update the vehicles that are in route
    in_route_veh_df, completed_veh_df = assign_in_route_to_next_edges(
        in_route_veh_df, edges_df, interval_length, current_time, completed_veh_df
    )
    # check that the row counts match
    match_row_counts('D', original_number_of_rows, in_queue_veh_df, in_route_veh_df, completed_veh_df, current_time)
    
    return in_route_veh_df, in_queue_veh_df, edges_df, completed_veh_df


def run_full_loop(od_matrix, weight=1, min_speed=5, for_all_road_types=False, 
                  start_time='05:00:00', end_time=None, interval_length='5s'):
    """
    Run the full traffic simulation loop.
    
    Args:
        weight: Weight parameter for congestion calculation (default=1)
        min_speed: Minimum speed parameter (default=5)
        for_all_road_types: Boolean indicating whether to apply to all road types (default=False)
        start_time: Start time of the simulation (default='05:00:00')
        end_time: End time of the simulation (default=None)
        interval_length: Length of the time interval (default='5s')
        
    Returns:
        Tuple of (edges_df, in_route_veh_df, completed_veh_df)
    """
    # (1) Initialize required dataframes
    in_queue_veh_df, edges_df = initial_dfs_creation(od_matrix)
    # check the shape of in_queue_veh_df
    original_number_of_rows = in_queue_veh_df.shape[0]
    print("original_number_of_rows:", in_queue_veh_df.shape[0])
    original_number_of_edges = edges_df.shape[0]
    
    # (2) Preparation regardless of end_time
    # (2a) Create initial in_route_veh_df and completed_veh_df
    in_route_veh_df = pd.DataFrame(
        columns=['person_id', 'route_links', 'current_link', 'remaining_time_on_current_link', 
                'interval_length', 'fully_adjusted_in_step_a']
    )
    completed_veh_df = pd.DataFrame(columns=['person_id', 'completion_time'])
    
    # (2b) Create network graph
    G = create_network_graph(edges_df)
    
    # (2c) Find the dispatch interval time (i.e. how frequently we are recording vehicles leaving their starting positions)
    dispatch_interval_time = find_queue_dispatch_frequency(in_queue_veh_df)

    # (3) Handle execution based on whether end_time is provided
    if end_time is not None: 
        print("end_time provided")
        # (3a) Create time_intervals_df
        time_intervals_df = create_time_interval_df(start_time, end_time, interval_length)
        
        # (3b) Run the loop with fixed intervals
        for current_time in time_intervals_df['time_interval']:
            in_route_veh_df, in_queue_veh_df, edges_df, completed_veh_df = process_single_time_step(
                current_time, in_route_veh_df, in_queue_veh_df, edges_df, G, 
                dispatch_interval_time, interval_length, weight, min_speed, 
                for_all_road_types, completed_veh_df, original_number_of_rows, original_number_of_edges
            )
    # (4) Handle execution when no end_time is provided
    else:
        print("end_time not provided. Looping until both in_route_veh_df AND queue_veh_df are empty")
        current_time = pd.to_datetime(start_time).time()

        while not in_route_veh_df.empty or not in_queue_veh_df.empty:
            in_route_veh_df, in_queue_veh_df, edges_df, completed_veh_df = process_single_time_step(
                current_time, in_route_veh_df, in_queue_veh_df, edges_df, G, 
                dispatch_interval_time, interval_length, weight, min_speed, 
                for_all_road_types, completed_veh_df, original_number_of_rows, original_number_of_edges
            )
            
            # update current_time
            current_time = create_next_time_unit(current_time, interval_length)

    return edges_df, in_route_veh_df, completed_veh_df
