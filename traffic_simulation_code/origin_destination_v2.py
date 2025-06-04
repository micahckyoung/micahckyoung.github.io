# the purpose of this code is to prepare the Origin-Destination Matrix

# v2 filters out one-way streets from the nodes dataset
# so if the node is two-way, it will be included
# this is important because I want nodes that are accessible from all directions

# import necessary packages
import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
import requests
import os
import zipfile
from scipy.spatial.distance import cdist
import random

# -----------------
# (1) Read the LODES & Tiger & Nodes data
# -----------------

# (1a) Read LODES

def create_LODES_path():
    read_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets')
    lodes_path = read_path + '/' + 'hi_od_main_JT00_2022.csv.gz'
    return lodes_path

# w_geocode is workplace
# h_geocode is home
def read_LODES(lodes_path):
    raw_lodes = pd.read_csv(lodes_path)
    raw_lodes['w_geocode'] = raw_lodes['w_geocode'].astype(str)
    raw_lodes['h_geocode'] = raw_lodes['h_geocode'].astype(str)
    # filter for Oahu
    raw_lodes = raw_lodes[raw_lodes['w_geocode'].str[:5] == '15003'].copy()
    raw_lodes = raw_lodes[raw_lodes['h_geocode'].str[:5] == '15003'].copy()
    print(f"Total LODES flow covered: {raw_lodes['S000'].sum()}")
    # add in CBG column
    raw_lodes['census_block_group_origin'] = raw_lodes['h_geocode'].str[:12]
    raw_lodes['census_block_group_destination'] = raw_lodes['w_geocode'].str[:12]
    lodes_unique_blocks = pd.concat([raw_lodes['w_geocode'], raw_lodes['h_geocode']]).unique()
    print(f"Number of unique census blocks across both columns: {len(lodes_unique_blocks)}")
    return raw_lodes, lodes_unique_blocks

# (1b) Read TIGER

# I'm working with LODES 8.3 which uses 2023 Census Blocks
def create_tiger_path():
    read_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets')
    tiger_path = read_path + '/' + 'tl_2023_15_tabblock20.zip'
    return tiger_path
    
def read_tiger(tiger_path):
    tiger_name = 'tl_2023_15_tabblock20.shp'
    raw_tiger = gpd.read_file(f"zip://{tiger_path}!{tiger_name}")
    # filter for Oahu
    raw_tiger = raw_tiger[raw_tiger['GEOID20'].str[:5] == '15003'].copy()
    return raw_tiger

# (1c) Read Nodes

def read_nodes_path():
    nodes_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'nodes.gpkg')
    return nodes_path

def read_nodes(nodes_path):
    nodes = gpd.read_file(nodes_path)
    return nodes

# (1d) Filter Nodes
# for this section, I need to read the original edges data (not the refined edges)

# def pull_twoway_nodes_from_osmx():
#     place_query = 'Oahu, Hawaii, USA'
#     G = ox.graph_from_place(place_query, network_type='drive')
#     edges_df = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
#     edges_df = edges_df.rename(columns={'u': 'Node1', 'v': 'Node2'})
#     edges_df['Node1'] = edges_df['Node1'].astype(str)
#     edges_df['Node2'] = edges_df['Node2'].astype(str)
#     # filter out one-way streets and collect all the nodes
#     twoway_edges = edges_df[edges_df['oneway'] == False].copy()
#     twoway_nodes = pd.concat([twoway_edges['Node1'], twoway_edges['Node2']]).unique().tolist()
#     return twoway_nodes

def read_edges_path():
    edges_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'processed', 'edges.gpkg')
    return edges_path

def read_edges(edges_path):
    edges_df = gpd.read_file(edges_path)
    return edges_df

# not super necessary now that I'm using a strongly connected network, but I'll keep it here for now
def filter_nodes_based_on_twoway(edges_df, nodes_gdf):
    # two-way nodes mean that the node is present as BOTH Node1 and Node2 
    node_1_list = edges_df['Node1'].unique().tolist()
    node_2_list = edges_df['Node2'].unique().tolist()
    in_node_1_and_node_2 = [node for node in node_1_list if node in node_2_list]
    print(f"Number of nodes before filtering for two-way streets: {nodes_gdf.shape[0]}")
    nodes_gdf = nodes_gdf[nodes_gdf['Node'].isin(in_node_1_and_node_2)].copy()
    print(f"Number of nodes after filtering for two-way streets: {nodes_gdf.shape[0]}")
    return nodes_gdf

# (1e) Merge data

# here what I'm doing is taking the unique CB's from the LODES data and merging it with the Tiger data
# later I will merge back to the full lodes data 

def filter_tiger(lodes_unique_blocks, tiger_gdf):
    lodes_gdf = tiger_gdf[tiger_gdf['GEOID20'].isin(lodes_unique_blocks)][['GEOID20', 'geometry']]
    return lodes_gdf

# -----------------
# (2) Tag Census Blocks to Nodes
# -----------------

# gather all border & internal nodes for all CB's
# for CB's with zero border nodes, find their bordering CB's and pull their nodes
# for remaining CB's with zero border nodes and zero neighbor border nodes, pull the nearest node
# pick the 'best' node out of the borders and internal nodes
# - the best node will be the node that is the closest to the concentration of the nodes (think clustering or mean center)

# again, note that lodes_gdf is only the unique CB's from the LODES data and not the full dataset
def gather_all_border_and_internal_nodes(lodes_gdf, nodes_gdf):
    # (1) reset indexes for clean joins
    lodes_gdf = lodes_gdf.reset_index(drop=True).copy()
    nodes_gdf = nodes_gdf.reset_index(drop=True).copy()
    # (2) ensure same CRS
    lodes_gdf = lodes_gdf.to_crs(nodes_gdf.crs)
    # (3) spatial join to get points within polygons
    # 'intersects' predicate to catch both interior and boundary points
    spatial_join = gpd.sjoin(nodes_gdf, lodes_gdf, predicate='intersects', how='left')
    spatial_join['Node'] = spatial_join['Node'].astype(str)
    # (4) group by polygon index and aggregate nodes into lists
    node_lists = spatial_join.groupby('index_right')['Node'].agg(list).reset_index()
    # (5) merge the node lists back to the original lodes_gdf
    result = lodes_gdf.merge(
        node_lists,
        left_index=True,
        right_on='index_right',
        how='left'
    ).rename(columns={'Node': 'node_list'})
    # (6) fill any polygons that didn't get any nodes with empty lists
    result['node_list'] = result['node_list'].fillna(result['node_list'].apply(lambda x: []))
    # (7) drop the temporary index column
    result = result.drop(columns=['index_right'])
    result = result.reset_index(drop=True)
    # (8) message for empty CB's
    empty_blocks = len(result[result['node_list'].str.len() == 0])
    print(f"There are {empty_blocks} out of {result.shape[0]} Census Blocks that don't border a single node")
    return result

def identify_neighbor_CBs(lodes_gdf):
    # (1) make a copy of input data
    lodes_gdf = lodes_gdf.reset_index(drop=True).copy()
    # (2) create spatial index for efficiency
    spatial_index = lodes_gdf.sindex
    # (3) create empty list to store neighbor pairs
    neighbor_list = []
    # (4) iterate through each polygon
    for idx, polygon in enumerate(lodes_gdf.geometry):
        # get potential neighbors using spatial index
        possible_matches_idx = list(spatial_index.intersection(polygon.bounds))
        possible_matches = lodes_gdf.iloc[possible_matches_idx]
        # remove self from potential matches
        possible_matches = possible_matches[possible_matches.index != idx]
        # check for actual intersection with boundaries
        actual_neighbors = possible_matches[possible_matches.geometry.touches(polygon)]
        # store the GEOID20s of neighbors
        neighbor_list.append(actual_neighbors['GEOID20'].tolist())
    # (5) add neighbor list as new column
    lodes_gdf['neighbor_cb_list'] = neighbor_list
    # (6) count and print number of isolated CBs
    isolated_cbs = len(lodes_gdf[lodes_gdf['neighbor_cb_list'].str.len() == 0])
    print(f"There are {isolated_cbs} out of {lodes_gdf.shape[0]} Census Blocks that don't border any other Census Blocks")
    return lodes_gdf

# this uses lodes_df output from the other two functions
def tag_neighbors_nodes(lodes_gdf):
    # (1) reset indexes for clean joins
    lodes_gdf = lodes_gdf.reset_index(drop=True).copy()
    # (2) create empty list to store neighbor nodes
    neighbor_nodes = []
    # (3) iterate through each Census Block
    for idx, row in lodes_gdf.iterrows():
        # get list of neighbor CBs
        neighbor_cbs = row['neighbor_cb_list']
        # initialize set for this CB's neighbor nodes
        all_neighbor_nodes = set()
        # get nodes from each neighbor CB
        for neighbor_geoid in neighbor_cbs:
            # find the neighbor's row and get its node list
            neighbor_nodes_list = lodes_gdf.loc[lodes_gdf['GEOID20'] == neighbor_geoid, 'node_list'].iloc[0]
            # add these nodes to our set
            all_neighbor_nodes.update(neighbor_nodes_list)
        # convert set back to list and store
        neighbor_nodes.append(list(all_neighbor_nodes))
    # (4) add new column to dataframe
    lodes_gdf['neighbor_nodes'] = neighbor_nodes
    # (5) add print statement
    num_neighbor_node_no_node_list = lodes_gdf[(lodes_gdf['neighbor_nodes'].str.len() != 0) & (lodes_gdf['node_list'].str.len() == 0)].shape[0]
    print(f"There are {num_neighbor_node_no_node_list} Census Blocks that don't have nodes, but that have neighbors that do")
    return lodes_gdf

def find_tagged_cbs(lodes_gdf):
    tagged_cbs = lodes_gdf[(lodes_gdf['neighbor_nodes'].str.len() > 0) | (lodes_gdf['node_list'].str.len() > 0)]
    print(f"There are {tagged_cbs.shape[0]} Census Blocks that do have nodes or neighbors with nodes")
    return tagged_cbs

# step 1: bordering nodes
# step 2: neighbor's bordering nodes
# this is run next
def find_final_untagged_cbs(lodes_gdf):
    untagged_cbs = lodes_gdf[(lodes_gdf['neighbor_nodes'].str.len() == 0) & (lodes_gdf['node_list'].str.len() == 0)]
    print(f"There are {untagged_cbs.shape[0]} Census Blocks that don't have any nodes or neighbors with nodes")
    return untagged_cbs

def tag_nearest_node(untagged_cbs, nodes_gdf):
    # (1) reset indexes for clean joins
    untagged_cbs = untagged_cbs.reset_index(drop=True).copy()
    nodes_gdf = nodes_gdf.reset_index(drop=True).copy()
    # (2) ensure same CRS
    untagged_cbs = untagged_cbs.to_crs(nodes_gdf.crs)
    # (3) function to find nearest node for a single polygon
    def find_nearest_node(polygon):
        # calculate distances to all nodes
        distances = [polygon.distance(node) for node in nodes_gdf.geometry]
        # get index of nearest node
        nearest_idx = distances.index(min(distances))
        # return the Node ID
        return nodes_gdf.iloc[nearest_idx]['Node']
    # (4) apply function to each polygon
    untagged_cbs['nearest_node'] = untagged_cbs.geometry.apply(find_nearest_node)
    # (5) print summary
    print(f"Tagged {len(untagged_cbs)} Census Blocks with their nearest node")
    return untagged_cbs

# goal: return lodes_gdf with 'median_node' column (single string)
# by median node, I mean that the node that is closest to the concentration of nodes
# I don't want outlier nodes to pull the node too far away -- i.e. I'm not looking for a mean function

# Uses scipy's cdist to calculate pairwise distances between all nodes in a cluster
# Finds the node with minimum total distance to all other nodes (geometric median)
# Handles edge cases like empty node lists and single-node clusters
def median_node_in_cluster(lodes_gdf, nodes_gdf, node_list_column):
    # (1) reset indexes for clean joins
    lodes_gdf = lodes_gdf.reset_index(drop=True).copy()
    nodes_gdf = nodes_gdf.reset_index(drop=True).copy()
    # (2) ensure same CRS
    lodes_gdf = lodes_gdf.to_crs(nodes_gdf.crs)
    # (3) create empty list to store median nodes
    median_nodes = []
    # (4) iterate through each Census Block
    for idx, row in lodes_gdf.iterrows():
        # (5) get list of neighbor nodes
        node_list = row[node_list_column]
        if not node_list:  # handle empty lists
            median_nodes.append(None)
            continue
        # (6) get subset of nodes for this block
        node_list_gdf = nodes_gdf[nodes_gdf['Node'].isin(node_list)]
        if len(node_list_gdf) == 0:
            median_nodes.append(None)
            continue

        # (7) if only one node, use it
        if len(node_list_gdf) == 1:
            median_nodes.append(node_list_gdf.iloc[0]['Node'])
            continue
        # (8) calculate pairwise distances between all nodes
        coords = np.column_stack((node_list_gdf.geometry.x, node_list_gdf.geometry.y))
        distances = cdist(coords, coords)
        # (9) sum distances from each point to all others
        total_distances = distances.sum(axis=1)
        # (10) point with minimum total distance is geometric median
        median_idx = np.argmin(total_distances)
        median_node = node_list_gdf.iloc[median_idx]['Node']
        median_nodes.append(median_node)
    # (11) add median nodes to lodes_gdf
    lodes_gdf['median_node'] = median_nodes
    return lodes_gdf

# I still need to create a final_node colum, add in untagged_cbs, create median_node, etc.

# main function for this section

def tag_cbs_to_nodes_main(lodes_gdf, nodes_gdf):
    # (1) gather all border and internal nodes
    lodes_gdf = gather_all_border_and_internal_nodes(lodes_gdf, nodes_gdf)
    # (2) identify neighbor CBs
    lodes_gdf = identify_neighbor_CBs(lodes_gdf)
    # (3) tag neighbor nodes
    lodes_gdf = tag_neighbors_nodes(lodes_gdf)

    # split into tagged and untagged CBs

    # (4) untagged CBs
    untagged_cbs = find_final_untagged_cbs(lodes_gdf)
    untagged_cbs = tag_nearest_node(untagged_cbs, nodes_gdf)

    # (5) tagged CBs
    tagged_cbs = find_tagged_cbs(lodes_gdf)
    # prioritize node_list over neighbor_nodes
    tagged_cbs['new_node_list'] = tagged_cbs.apply(lambda x: x['node_list'] if len(x['node_list']) > 0 else x['neighbor_nodes'], axis=1)
    if tagged_cbs[tagged_cbs['new_node_list'].str.len() > 0].shape[0] == tagged_cbs.shape[0]:
        print("All tagged CBs have nodes")
    tagged_cbs = median_node_in_cluster(tagged_cbs, nodes_gdf, 'new_node_list').copy()

    # (6) concatenate tagged and untagged CBs back together
    lodes_gdf = pd.concat([tagged_cbs, untagged_cbs])
    # prioritize nearest_node over median_node
    lodes_gdf['final_node_assignment'] = lodes_gdf.apply(lambda x: x['nearest_node'] if pd.isnull(x['median_node']) else x['median_node'], axis=1)
    return lodes_gdf

# -----------------
# (3) Pull ACS departure time
# -----------------

def read_acs_api_path():
    acs_api_path = os.path.join(os.path.dirname(os.getcwd()), 'api_keys/personal_census_API_key_241211.txt')
    return acs_api_path

def read_acs_api_key(acs_api_path):
    with open(acs_api_path, 'r') as file:
        acs_api_key = file.read()
    return acs_api_key

def get_acs_time_leaving_for_work(api_key, year='2022'):
    """
    Fetches ACS data about when people leave for work by Census Block Group,
    including both time intervals and response rates.
    
    Args:
        api_key (str): Census API key
        year (str): Year of data to fetch, defaults to '2022'
    
    Returns:
        pandas.DataFrame: DataFrame containing departure times and response rates by CBG
    """
    base_url = "https://api.census.gov/data"
    
    # Time departure variables (B08302 series)
    time_vars = [
        'B08302_001E',  # Total
        'B08302_002E',  # 12:00 a.m. to 4:59 a.m.
        'B08302_003E',  # 5:00 a.m. to 5:29 a.m.
        'B08302_004E',  # 5:30 a.m. to 5:59 a.m.
        'B08302_005E',  # 6:00 a.m. to 6:29 a.m.
        'B08302_006E',  # 6:30 a.m. to 6:59 a.m.
        'B08302_007E',  # 7:00 a.m. to 7:29 a.m.
        'B08302_008E',  # 7:30 a.m. to 7:59 a.m.
        'B08302_009E',  # 8:00 a.m. to 8:29 a.m.
        'B08302_010E',  # 8:30 a.m. to 8:59 a.m.
        'B08302_011E',  # 9:00 a.m. to 9:59 a.m.
        'B08302_012E',  # 10:00 a.m. to 10:59 a.m.
        'B08302_013E',  # 11:00 a.m. to 11:59 a.m.
        'B08302_014E',  # 12:00 p.m. to 3:59 p.m.
        'B08302_015E'   # 4:00 p.m. to 11:59 p.m.
    ]
    
    # Response rate variables
    quality_vars = [
        'B99083_001E',  # Total workers
        'B99083_003E'   # Workers with allocated departure time
    ]
    
    # Combine all variables
    variables = time_vars + quality_vars
    
    # Construct the API URL for Honolulu County (FIPS: 15003)
    url = f"{base_url}/{year}/acs/acs5"
    params = {
        'get': ','.join(['NAME'] + variables),
        'for': 'block group:*',
        'in': 'state:15 county:003',
        'key': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Convert response to DataFrame
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Convert numeric columns to float
        numeric_columns = variables
        df[numeric_columns] = df[numeric_columns].astype(float)

        # Create census_block_group identifier
        df['census_block_group'] = (
            df['state'].str.zfill(2) + 
            df['county'].str.zfill(3) + 
            df['tract'].str.zfill(6) + 
            df['block group'].str.zfill(1)
        )
        
        # Calculate response rate
        df['response_rate'] = (df['B08302_001E'] / df['B99083_001E'] * 100).round(1)
        
        # First rename the columns
        new_names = {
            'B08302_001E': 'total_workers',
            'B08302_002E': 'leave_12am_to_459am',
            'B08302_003E': 'leave_5am_to_529am',
            'B08302_004E': 'leave_530am_to_559am',
            'B08302_005E': 'leave_6am_to_629am',
            'B08302_006E': 'leave_630am_to_659am',
            'B08302_007E': 'leave_7am_to_729am',
            'B08302_008E': 'leave_730am_to_759am',
            'B08302_009E': 'leave_8am_to_829am',
            'B08302_010E': 'leave_830am_to_859am',
            'B08302_011E': 'leave_9am_to_959am',
            'B08302_012E': 'leave_10am_to_1059am',
            'B08302_013E': 'leave_11am_to_1159am',
            'B08302_014E': 'leave_12pm_to_359pm',
            'B08302_015E': 'leave_4pm_to_1159pm',
            'B99083_001E': 'total_eligible_workers',
            'B99083_003E': 'workers_with_departure_time'
        }
        df = df.rename(columns=new_names)
        
        # IMPORTANT: Get time columns from the renamed DataFrame columns
        time_columns = [col for col in df.columns if col.startswith('leave_')]
        
        # Calculate percentages for each time interval
        for col in time_columns:
            pct_col = f'{col}_pct'
            df[pct_col] = (df[col] / df['total_workers'] * 100).round(1)
        
        # Reorder columns
        geo_cols = ['NAME', 'census_block_group', 'state', 'county', 'tract', 'block group']
        metadata_cols = ['total_workers', 'total_eligible_workers', 'workers_with_departure_time', 'response_rate']
        count_cols = time_columns
        pct_cols = [f'{col}_pct' for col in time_columns]
        
        df = df[geo_cols + metadata_cols + count_cols + pct_cols]
        # convert NaN to 0
        df = df.fillna(0)
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# -----------------
# (4) Pull ACS transportation mode
# -----------------

def get_acs_transportation_modes(api_key, year='2022'):
    """
    Fetches ACS data about transportation modes used for commuting by Census Block Group,
    including both counts and response rates.
    
    Args:
        api_key (str): Census API key
        year (str): Year of data to fetch, defaults to '2022'
    
    Returns:
        pandas.DataFrame: DataFrame containing transportation modes and response rates by CBG
    """
    base_url = "https://api.census.gov/data"
    
    # Transportation mode variables (B08301 series)
    transport_vars = [
        'B08301_001E',  # Total workers 16 and over
        'B08301_003E',  # Car, truck, or van - drove alone
        'B08301_004E',  # Carpooled total
        'B08301_005E',  # In 2-person carpool
        'B08301_006E',  # In 3-person carpool
        'B08301_007E',  # In 4-person carpool
        'B08301_008E',  # In 5- or 6-person carpool
        'B08301_009E',  # In 7-or-more-person carpool
        'B08301_010E',  # Public transportation (excluding taxicab)
        'B08301_018E',  # Bicycle
        'B08301_019E',  # Walked
        'B08301_020E',  # Other means
        'B08301_021E'   # Worked from home
    ]
    
    # Response rate variables
    quality_vars = [
        'B99081_001E',  # Total workers
        'B99081_002E'   # Workers with allocated means of transportation
    ]
    
    # Combine all variables
    variables = transport_vars + quality_vars
    
    # Construct the API URL for Honolulu County (FIPS: 15003)
    url = f"{base_url}/{year}/acs/acs5"
    params = {
        'get': ','.join(['NAME'] + variables),
        'for': 'block group:*',
        'in': 'state:15 county:003',
        'key': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Convert response to DataFrame
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Convert numeric columns to float
        numeric_columns = variables
        df[numeric_columns] = df[numeric_columns].astype(float)

        # Create census_block_group identifier
        df['census_block_group'] = (
            df['state'].str.zfill(2) + 
            df['county'].str.zfill(3) + 
            df['tract'].str.zfill(6) + 
            df['block group'].str.zfill(1)
        )
        
        # Calculate response rate
        df['response_rate'] = (df['B08301_001E'] / df['B99081_001E'] * 100).round(1)
        
        # Rename the columns
        new_names = {
            'B08301_001E': 'total_workers',
            'B08301_003E': 'drove_alone',
            'B08301_004E': 'carpooled',
            'B08301_005E': 'carpool_2person',
            'B08301_006E': 'carpool_3person',
            'B08301_007E': 'carpool_4person',
            'B08301_008E': 'carpool_5_6person',
            'B08301_009E': 'carpool_7plus_person',
            'B08301_010E': 'public_transit',
            'B08301_018E': 'bicycle',
            'B08301_019E': 'walked',
            'B08301_020E': 'other_means',
            'B08301_021E': 'worked_from_home',
            'B99081_001E': 'total_eligible_workers',
            'B99081_002E': 'workers_with_transport_mode'
        }
        df = df.rename(columns=new_names)
        
        # Calculate sums for validation
        df['carpool_sum'] = (df['carpool_2person'] + df['carpool_3person'] + 
                            df['carpool_4person'] + df['carpool_5_6person'] + 
                            df['carpool_7plus_person'])
        
        df['car_total'] = df['drove_alone'] + df['carpooled']
        
        # Simple validation prints
        print(f"Carpool components sum to total for all CBGs: {(abs(df['carpool_sum'] - df['carpooled']) < 1).all()}")
        print(f"Car total <= total workers for all CBGs: {(df['car_total'] <= df['total_workers']).all()}")
        
        # Define main transportation columns and carpool breakdown columns
        main_transport_columns = ['drove_alone', 'carpooled', 'public_transit', 
                                'walked', 'bicycle', 'other_means', 'worked_from_home']
        
        carpool_columns = ['carpool_2person', 'carpool_3person', 'carpool_4person',
                          'carpool_5_6person', 'carpool_7plus_person']
        
        # Calculate percentages
        for col in main_transport_columns:
            pct_col = f'{col}_pct'
            df[pct_col] = (df[col] / df['total_workers'] * 100).round(1)
        
        for col in carpool_columns:
            pct_col = f'{col}_pct'
            df[pct_col] = (df[col] / df['carpooled'] * 100).round(1)
        
        # Reorder columns
        geo_cols = ['NAME', 'census_block_group', 'state', 'county', 'tract', 'block group']
        metadata_cols = ['total_workers', 'total_eligible_workers', 'workers_with_transport_mode', 'response_rate']
        count_cols = main_transport_columns + carpool_columns
        pct_cols = [f'{col}_pct' for col in main_transport_columns + carpool_columns]
        
        df = df[geo_cols + metadata_cols + count_cols + pct_cols]
        # convert NaN to 0
        df = df.fillna(0)
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# -----------------
# (5) Create final car dataset with ACS and LODES data
# -----------------

# problem: over 75% of the Origin-Destination pairs cover only 1 person
# this means that when I apply the ACS filter percentages, I get a tiny fraction of a car for each O-D row
# I will need to take the CBG percentages and multiply them by the total number of jobs for that CBG
# -- i.e. aggregate lodes_df on origin CBG to get the total jobs for that CBG
# once I multiply the pct to that CBG total, I can then randomly select jobs from the existing CBG's
# that way, I can work with whole numbers

# I can create a dataset where every row is an individual person (S000 = 1 in every case)
# I can create a unique identifier for these rows, then draw a random sample of the rows

# create a dataset where each row is a car
# include a column for time of departure with options from ACS
# have everything check out to the total number of cars on the road

# -----------------
# (5a) determine the number of cars on the road for each CBG


# I will multiply this dataset by the ACS transportation mode percentages to get a number of cars per CBG
def num_jobs_per_CBG(lodes_df):
    jobs_per_cbg_df = lodes_df.groupby('census_block_group_origin')['S000'].sum().reset_index()
    jobs_per_cbg_df = jobs_per_cbg_df.rename(columns = {'S000': 'total_jobs'})
    return jobs_per_cbg_df

def prepare_acs_transportation_mode(acs_transportation_mode_df):
    # if 2 people carpool together, one of those two people is worth half of that carpool percentage
    # (1) create new cars to humans ratio column
    acs_transportation_mode_df['total_cars_on_road'] = (
        acs_transportation_mode_df['drove_alone'] + 
        (acs_transportation_mode_df['carpool_2person'] / 2) + 
        (acs_transportation_mode_df['carpool_3person'] / 3) +
        (acs_transportation_mode_df['carpool_4person'] / 4) +
        (acs_transportation_mode_df['carpool_5_6person'] / 5.5) +
        (acs_transportation_mode_df['carpool_7plus_person'] / 7.5)
        )
    acs_transportation_mode_df['car_to_worker_ratio'] = (acs_transportation_mode_df['total_cars_on_road'] / acs_transportation_mode_df['total_workers']) * 100
    # (2) if car_to_worker_ratio is NaN or 0, replace with the median
    median_car_to_worker_ratio = acs_transportation_mode_df['car_to_worker_ratio'].median()
    acs_transportation_mode_df['car_to_worker_ratio'] = (
        acs_transportation_mode_df['car_to_worker_ratio']
        .fillna(median_car_to_worker_ratio)
        .replace(0, median_car_to_worker_ratio)
        )
    return acs_transportation_mode_df

def merge_jobs_and_acs_transportation(jobs_per_cbg_df, acs_transportation_mode_df):
    merged_df = pd.merge(jobs_per_cbg_df, acs_transportation_mode_df[['census_block_group', 'car_to_worker_ratio']], how = 'left', 
                         left_on = 'census_block_group_origin', right_on = 'census_block_group')
    merged_df = merged_df.drop(columns = ['census_block_group'])
    # add in a column for the number of cars on the road
    merged_df['total_cars_on_road'] = (merged_df['total_jobs'] * (merged_df['car_to_worker_ratio'] / 100)).round(0).astype(int)
    return merged_df # also can be known as cars_per_cbg_df

# -----------------
# (5b) expand the LODES dataset and randomly sample each CBG based on the number of cars on the road

# expand the LODES dataset to include each individual
# I'm expanding the number of rows by num_people and then creating a new column for the unique identifier for each row
def expand_lodes_rows(lodes_df):
    # (1) create helper columns
    lodes_df['num_people'] = lodes_df['S000']
    lodes_df['origin_destination_unique_id'] = lodes_df['h_geocode'] + '_' + lodes_df['w_geocode']
    # (2) create list of indices to repeat based on num_people
    indices_to_repeat = np.repeat(lodes_df.index.values, lodes_df['num_people'])
    # (3) expand the dataframe by repeating rows
    expanded_df = lodes_df.loc[indices_to_repeat].copy()
    # (4) create unique identifier for each person
    expanded_df['person_id'] = range(len(expanded_df))
    # (5) create full unique identifier combining origin-destination and person
    expanded_df['unique_id'] = expanded_df['origin_destination_unique_id'] + '_' + expanded_df['person_id'].astype(str)
    # (6) reset index and clean up
    expanded_df = expanded_df.reset_index(drop=True)
    expanded_df = expanded_df.drop(columns=['num_people'])
    return expanded_df

# I'm randomly selecting rows from expand_lodes_rows based on the number of cars in that CBG
# the random selection is necessary because these rows will have different origin and destination CB's
# i.e. if cars_per_cbg_df says that there are 10 total_cars_on_road for CBG 001, 
# then I am only selecting 10 rows with that CBG from expanded_df
def random_selection_expanded_lodes_rows(expanded_df, cars_per_cbg_df, random_seed):
    # (1) create a dictionary with the number of cars in each CBG
    cars_dict = dict(zip(cars_per_cbg_df['census_block_group_origin'], cars_per_cbg_df['total_cars_on_road']))
    # (2) create a list to store selected rows
    selected_rows = []
    # (3) iterate through each unique CBG -- note: I only care about the expanded_df CBG's because those are the ones I have data for
    # I also only care about origin because the ACS data is based on origin (not destination)
    for cbg in expanded_df['census_block_group_origin'].unique():
        # (4) get rows for this CBG
        cbg_rows = expanded_df[expanded_df['census_block_group_origin'] == cbg]
        # (5) get number of cars for this CBG, default to 0 if not found
        num_cars = cars_dict.get(cbg, 0)
        # (6) if there are cars and rows to sample from
        if num_cars > 0 and len(cbg_rows) > 0:
            # (7) sample with replacement if we need more rows than available
            replace = num_cars > len(cbg_rows)
            # (8) sample the appropriate number of rows -- min function handles cases where there are more cars than rows (this shouldn't happen however)
            sampled_rows = cbg_rows.sample(n=min(num_cars, len(cbg_rows)), replace=replace, random_state=random_seed)
            selected_rows.append(sampled_rows)
    # (9) combine all selected rows
    final_df = pd.concat(selected_rows, ignore_index=True).reset_index(drop=True) # also named expanded_filtered_df
    return final_df

# -----------------
# (5c) prepare ACS time of departure data and tag time of departure to the expanded data

# it's weird to fill with the median of a percentage column, but the normalization fixes things.
# an alternative path could have been to take the median of the raw columns and the total_workers and then recalculate pct 
def prepare_acs_departure_time(acs_time_leaving_df):
    # if 'total_workers' is zero, then fill all the pct columns with median pct and normalize the percentages so that they add to 1
    # (1) identify all time percentage columns
    time_pct_columns = [col for col in acs_time_leaving_df.columns if (col.startswith('leave_') & col.endswith('_pct'))]
    # (2) identify rows with zero or NaN total workers
    invalid_worker_mask = (acs_time_leaving_df['total_workers'] == 0) | (acs_time_leaving_df['total_workers'].isna())
    # (3) calculate median percentages for each time column
    median_pcts = acs_time_leaving_df.loc[~invalid_worker_mask, time_pct_columns].median()
    # (4) fill zero worker rows with median values
    acs_time_leaving_df.loc[invalid_worker_mask, time_pct_columns] = median_pcts
    # (5) fill any remaining NaN values with median percentages
    acs_time_leaving_df[time_pct_columns] = acs_time_leaving_df[time_pct_columns].fillna(median_pcts)
    # (6) ensure percentages sum to 100 for each row
    row_sums = acs_time_leaving_df[time_pct_columns].sum(axis=1)
    # (7) handle any zero row sums to avoid division by zero
    row_sums = row_sums.replace(0, 1)  # replace zeros with 1 to avoid division by zero
    # (8) normalize percentages
    for col in time_pct_columns:
        acs_time_leaving_df[col] = acs_time_leaving_df[col] / row_sums * 100
    # (9) select final columns
    acs_time_leaving_df = acs_time_leaving_df[['census_block_group'] + time_pct_columns]
    return acs_time_leaving_df

# fixing the rounding errors makes this function more complicated
# by round the number of cars for each time period, I can get a total that is off by a few cars
def convert_acs_departure_time_to_cars(acs_time_leaving_df, cars_per_cbg_df):
    # (1) merge datasets
    merged_df = pd.merge(cars_per_cbg_df[['census_block_group_origin', 'total_cars_on_road']], 
                         acs_time_leaving_df, how='left', left_on='census_block_group_origin', 
                         right_on='census_block_group')
    # (2) create new columns for the number of cars on the road for each time interval
    time_pct_columns = [col for col in acs_time_leaving_df.columns if (col.startswith('leave_') & col.endswith('_pct'))]
    for col in time_pct_columns:
        col_strip_pct = col.replace('_pct', '')
        merged_df[f"cars_{col_strip_pct}"] = (merged_df['total_cars_on_road'] * (merged_df[col] / 100)).round(0).astype(int)
    # (3) get the car columns and calculate totals
    car_columns = [col for col in merged_df.columns if col.startswith('cars_')]
    merged_df['calculated_total'] = merged_df[car_columns].sum(axis=1)
    # (4) fix any rounding errors -- rounding causes the total to be off in some cases
    for idx in merged_df.index:
        difference = merged_df.loc[idx, 'total_cars_on_road'] - merged_df.loc[idx, 'calculated_total']
        if difference != 0:
            # find the largest car column value for this row
            largest_car_col = max(car_columns, key=lambda x: merged_df.loc[idx, x])
            # adjust the largest column by the difference
            merged_df.loc[idx, largest_car_col] += difference
    # (5) final check to ensure totals match
    merged_df['final_total'] = merged_df[car_columns].sum(axis=1)
    if (merged_df['final_total'] == merged_df['total_cars_on_road']).all():
        print("All totals from function convert_acs_departure_time_to_cars() match")
    # (5) select final columns
    merged_df = merged_df[['census_block_group'] + car_columns]
    return merged_df

# I'm writing a function similar to random_selection_expanded_lodes_rows() but more complicated.
# Rather than pulling a sample, I'm randomly tagging each row with a time of departure based on
# the number of cars for that CBG AND time of departure.
# I'm creating the exact distribution of time periods needed, then shuffling those directly

def tag_time_of_departure(expanded_filtered_df, acs_time_leaving_df, random_seed):
    # (1) collect the time of departure columns
    departure_car_columns = [col for col in acs_time_leaving_df.columns if col.startswith('cars_')]
    # (2) create copy of input dataframe to modify
    result_df = expanded_filtered_df.copy()
    # (3) iterate through each CBG
    for cbg in expanded_filtered_df['census_block_group_origin'].unique():
        # (4) get rows for this CBG
        cbg_rows = expanded_filtered_df[expanded_filtered_df['census_block_group_origin'] == cbg]
        # (5) get car counts for each time period for this CBG
        if cbg in acs_time_leaving_df['census_block_group'].values:
            car_counts = acs_time_leaving_df[acs_time_leaving_df['census_block_group'] == cbg][departure_car_columns].iloc[0]
        else:
            continue
        # (6) create list of time periods based on car counts
        time_periods = []
        for col in departure_car_columns:
            time_period = col.replace('cars_', '')
            cars_in_period = int(car_counts[col])
            time_periods.extend([time_period] * cars_in_period)
        # (7) shuffle the time periods list
        random.Random(random_seed).shuffle(time_periods)
        # (8) assign shuffled time periods to rows
        if len(time_periods) > 0:
            result_df.loc[cbg_rows.index[:len(time_periods)], 'departure_time'] = time_periods[:len(cbg_rows)]
    # (9) fill any unassigned rows with a default time period
    result_df['departure_time'] = result_df['departure_time'].fillna(departure_car_columns[0].replace('cars_', ''))
    result_df = result_df[['person_id', 'census_block_group_origin', 'w_geocode', 'h_geocode', 'departure_time']].copy()
    return result_df

# main function for step 5
def step5_main_acs_preparation(lodes_df, acs_transportation_mode_df, acs_time_leaving_df, random_seed):
    # (1) calculate number of jobs per CBG
    jobs_per_cbg_df = num_jobs_per_CBG(lodes_df)
    # (2) prepare ACS transportation mode data
    acs_transportation_mode_df = prepare_acs_transportation_mode(acs_transportation_mode_df)
    # (3) merge jobs and ACS transportation mode data
    cars_per_cbg_df = merge_jobs_and_acs_transportation(jobs_per_cbg_df, acs_transportation_mode_df)
    # (4) expand LODES rows and randomly sample based on number of cars
    expanded_df = expand_lodes_rows(lodes_df)
    expanded_filtered_df = random_selection_expanded_lodes_rows(expanded_df, cars_per_cbg_df, random_seed)
    # (5) prepare ACS time of departure data and tag time of departure to the expanded data
    acs_time_leaving_df = prepare_acs_departure_time(acs_time_leaving_df)
    acs_time_leaving_df = convert_acs_departure_time_to_cars(acs_time_leaving_df, cars_per_cbg_df)
    final_5 = tag_time_of_departure(expanded_filtered_df, acs_time_leaving_df, random_seed)
    return final_5

# -----------------
# (6) Split the 30min intervals into 5min intervals
# -----------------

# the reason why this section is necessary is that I don't want to overload the origin nodes
# compressing all people into a single 30min interval will cause congestion in the beginning of their trip
# this congestion is unrealistic because not everyone leaves at the exact 30min mark
# instead, I'm assuming that people leave uniformly with that 30min interval

# this code adds a column called '5min_designation'
# it uniformly assigns each row to a 5min interval within a single departure_time
# within a departure time, who gets assigned to which 5min interval is random
# assignment is based on BOTH CBG and departure time
def tag_5min_intervals(car_to_departure, random_seed):
    # (1) filter for morning rush hour traffic departure times
    # note: I am going to keep 9am to 9:59am, BUT this is an hour increment rather than 30min
    cols_filter_out = ['leave_12am_to_459am', 'leave_10am_to_1059am', 'leave_11am_to_1159am', 'leave_12pm_to_359pm', 'leave_4pm_to_1159pm']
    car_filtered = car_to_departure[~car_to_departure['departure_time'].isin(cols_filter_out)]
    # (2) split 'leave_9am_to_959am' evenly within a CBG. Make sure assignment to either 'leave_9am_to_929am' or 'leave_930am_to_959am' is random
    result_df = car_filtered.copy()
    for cbg in car_filtered[car_filtered['departure_time'] == 'leave_9am_to_959am']['census_block_group_origin'].unique():
        mask = (car_filtered['census_block_group_origin'] == cbg) & (car_filtered['departure_time'] == 'leave_9am_to_959am')
        cbg_rows = car_filtered[mask]
        n_rows = len(cbg_rows)
        # Randomly assign half to first 30min block, half to second 30min block
        # // is floor division which means that the sum of assigned rows equals the original number of rows
        first_half = ['leave_9am_to_929am'] * (n_rows // 2)
        second_half = ['leave_930am_to_959am'] * (n_rows - len(first_half))
        new_times = first_half + second_half
        random.Random(random_seed).shuffle(new_times)
        result_df.loc[cbg_rows.index, 'departure_time'] = new_times
    # (3) initialize the 5min_designation column
    result_df['5min_designation'] = None
    # (4) iterate through each unique combination of CBG and departure_time
    for (cbg, dep_time) in result_df.groupby(['census_block_group_origin', 'departure_time']).groups:
    # (5) get rows for this CBG and departure time
        mask = (result_df['census_block_group_origin'] == cbg) & (result_df['departure_time'] == dep_time)
        rows = result_df[mask]
    # (6) create list of 5-minute intervals
        intervals = []
        for i in range(6):  # 6 intervals of 5 minutes each
            interval = f"{dep_time}_interval_{i+1}"
            intervals.extend([interval] * (len(rows) // 6 + (1 if i < len(rows) % 6 else 0)))
    # (7) shuffle the intervals
        random.Random(random_seed).shuffle(intervals)
    # (8) assign intervals to rows
        result_df.loc[rows.index, '5min_designation'] = intervals
    return result_df[['person_id', 'census_block_group_origin', 'w_geocode', 'h_geocode','departure_time', '5min_designation']].copy()

# create a time interval index
# for example: 5:00am is 1 and 5:05am is 2
def create_time_interval_index(time_interval_df):
    # (1) create ordered list of departure times
    departure_times = [
        'leave_5am_to_529am',
        'leave_530am_to_559am',
        'leave_6am_to_629am',
        'leave_630am_to_659am',
        'leave_7am_to_729am',
        'leave_730am_to_759am',
        'leave_8am_to_829am',
        'leave_830am_to_859am',
        'leave_9am_to_929am',
        'leave_930am_to_959am'
    ]
    # (2) create mapping programmatically
    time_mapping = {time: 1 + (6 * i) for i, time in enumerate(departure_times)}
    # (3) extract interval number from 5min_designation
    time_interval_df['interval_num'] = time_interval_df['5min_designation'].str.extract(r'interval_(\d+)').astype(int)
    # (4) calculate final index
    time_interval_df['time_index'] = time_interval_df.apply(
        lambda row: time_mapping[row['departure_time']] + row['interval_num'] - 1, 
        axis=1
    )
    # (5) drop intermediate column and return
    return time_interval_df.drop('interval_num', axis=1)

# -----------------
# (7) Main functions
# -----------------

def main_origin_destination_matrix_function(random_seed=42):
    # (Step 1) read the necessary data
    lodes_path = create_LODES_path()
    lodes_df, lodes_unique_blocks = read_LODES(lodes_path)
    tiger_path = create_tiger_path()
    raw_tiger = read_tiger(tiger_path)
    node_path = read_nodes_path()
    nodes_gdf = read_nodes(node_path)
    # filter for two-way nodes
    edges_path = read_edges_path()
    edges_df = read_edges(edges_path)
    nodes_gdf = filter_nodes_based_on_twoway(edges_df, nodes_gdf)

    lodes_gdf = filter_tiger(lodes_unique_blocks, raw_tiger)
    # (Step 2) tag Census Blocks to nodes
    lodes_gdf = tag_cbs_to_nodes_main(lodes_gdf, nodes_gdf)
    # (Step 3 & 4) pull ACS data
    api_key_path = read_acs_api_path()
    api_key = read_acs_api_key(api_key_path)
    acs_transportation_mode_df = get_acs_transportation_modes(api_key)
    acs_time_leaving_df = get_acs_time_leaving_for_work(api_key)
    # (Step 5) prepare ACS and lodes_df data so that each row is an individual car tagged to a departure time
    car_to_departure = step5_main_acs_preparation(lodes_df, acs_transportation_mode_df, acs_time_leaving_df, random_seed)
    # (Step 6) split to 5min intervals and create an index for each time interval
    car_to_departure_5min = tag_5min_intervals(car_to_departure, random_seed)
    car_to_departure_final = create_time_interval_index(car_to_departure_5min)

    # merge the car_to_departure with the lodes_gdf
    car_to_departure_final = pd.merge(car_to_departure_final, lodes_gdf[['GEOID20', 'final_node_assignment']], how='left', left_on='h_geocode', right_on='GEOID20')
    car_to_departure_final.rename(columns = {'final_node_assignment': 'origin_node'}, inplace=True)
    car_to_departure_final.drop(columns = 'GEOID20', inplace=True)
    car_to_departure_final = pd.merge(car_to_departure_final, lodes_gdf[['GEOID20', 'final_node_assignment']], how='left', left_on='w_geocode', right_on='GEOID20')
    car_to_departure_final.rename(columns = {'final_node_assignment': 'destination_node'}, inplace=True)
    car_to_departure_final.drop(columns = 'GEOID20', inplace=True)

    # drop rows where the origin_node and the destination_node are the same
    print(f"Number of rows before dropping same origin and destination: {len(car_to_departure_final)}")
    car_to_departure_final = car_to_departure_final[car_to_departure_final['origin_node'] != car_to_departure_final['destination_node']].copy()
    print(f"Number of rows after dropping same origin and destination: {len(car_to_departure_final)}")
    return car_to_departure_final
