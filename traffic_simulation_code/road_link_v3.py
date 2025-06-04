# the purpose of this code is to prepare the road link datasets (both nodes and edges)
# v3 as a more advanced local roads filter -- I don't want to create islands in my network
# I only want to remove local roads that aren't the single connection between other road types

# import necessary packages
import networkx as nx
import osmnx as ox
import re
import pandas as pd
import numpy as np
import ast
# import networkx as nx
# note: these bootstrapped packages are used to calculate confidence intervals even though I only use median
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


# -----------------
# (1) Basic preparation of edges
# -----------------

# fix the data types within the 'highway' column of edges_df
def categorize_highway(highway_type):
    try:
        # Handle NaN
        if isinstance(highway_type, float) and np.isnan(highway_type):
            return highway_type
            
        # Handle bracketed strings
        if isinstance(highway_type, str) and '[' in highway_type:
            # Extract first value from bracketed list
            highway_type = highway_type.strip('[]').split(',')[0].strip()
            return highway_type
            
        # Handle list input
        if isinstance(highway_type, list):
            return highway_type[0]  # Take first value instead of second
            
        # Return the highway type as is (preserving link types)
        return str(highway_type)
        
    except (ValueError, IndexError):
        return np.nan  # Return NaN if conversion fails

def clean_max_speed(speed_str):
    try:
        if isinstance(speed_str, float) and np.isnan(speed_str):
            return speed_str
            
        speed_str = str(speed_str)
        speed_str = speed_str.replace('mph', '').strip()
        
        if speed_str.startswith('['):
            speeds = [int(x) for x in speed_str if x.isdigit()]
            return max(speeds)
        
        # If it's a single number, convert to int
        return int(''.join(x for x in speed_str if x.isdigit()))
    except ValueError:
        return speed_str  # Return original if conversion fails

def clean_lanes(lanes_value):
    try:
        # Handle NaN
        if isinstance(lanes_value, float) and np.isnan(lanes_value):
            return lanes_value
        # Convert to string
        lanes_str = str(lanes_value)
        # Handle bracketed values
        if '[' in lanes_str:
            # Extract digits and take first number
            nums = [int(x) for x in lanes_str if x.isdigit()]
            return nums[0] if nums else np.nan
        # If it's a single number, convert to int
        return int(''.join(x for x in lanes_str if x.isdigit()))
    except (ValueError, IndexError):
        return np.nan  # Return NaN if conversion fails

# write out summary stats
def understand_edges(edges_df):
    print(f"total number of edges: {edges_df.shape[0]}")
    print(f"number of edges missing 'highway': {edges_df['highway'].isnull().sum()}")
    print(f"number of edges missing 'lanes': {edges_df['lanes'].isnull().sum()}")
    print(f"number of edges missing 'maxspeed': {edges_df['maxspeed'].isnull().sum()}")
    return

# -----------------
# (2) Filling in missing lanes and speed
# -----------------

# these confidence interval functions are not necessary, I only pull median
# BUT they were helpful when I was trying to understand the importance of 'highway' category

def adjust_sig_level(sig_level):
	if sig_level < .5:
		sig_level = 1 - sig_level
	return(sig_level)

def compute_bootstrap_conf_intervals(series, sig_level, n_simulations=1000):
	# Anticipate sig_level = .95, not .05
	sig_level = adjust_sig_level(sig_level)
	""" Compute confidence intervals using bootstrapping. """
	res = bs.bootstrap(series, bs_stats.mean, alpha=1-sig_level, num_iterations=n_simulations)
	return res.value, res.lower_bound, res.upper_bound

def add_statistics(grouped, metric, ci_method="bootstrap", sig_level=0.95, min_sample_size=5, trunc_bottom = None, trunc_top = None, **kwargs):
	""" Compute various statistics for a given metric in the dataset. """
	# Compute median and bootstrapped confidence intervals.
	group_stats = []
	for name, group in grouped:
		vals = group[metric].values
		vals = vals[~pd.isna(vals)]
		if not len(vals):
			median, mean, lower, upper = 0.0, 0.0, 0.0, 0.0
		elif ci_method == "bootstrap":
			mean, lower, upper = compute_bootstrap_conf_intervals(vals, sig_level)
			median = np.median(vals)
		# elif ci_method == "beta":
		# 	mean, lower, upper = compute_beta_conf_intervals(vals, sig_level)
		# 	mean, lower, upper = 100 * mean, 100 * lower, 100 * upper
		# 	median = np.median(vals)
		# elif ci_method == "t":
		# 	mean, lower, upper = compute_t_conf_intervals(vals, sig_level)
		# 	median = np.median(vals)
		else:
			median, mean, lower, upper = np.median(vals), vals.mean(), vals.mean(), vals.mean()
		if trunc_bottom is not None:
			lower = max([lower, trunc_bottom])
		if trunc_top is not None:
			upper = min([upper, trunc_top])
		group_stats.append({
			'pivot': name, 
			'mean': mean,
			'lower_bound': lower,
			'upper_bound': upper,
			'lower_diff': mean - lower,
			'upper_diff': upper - mean,
			'sample_size': vals.shape[0],
			'median': median,
			'count': group.shape[0]
		})
	grouped = pd.DataFrame(group_stats)
	if min_sample_size > 0:
		grouped.at[grouped[grouped['sample_size'] < min_sample_size].index, grouped.columns != 'pivot'] = None
	return grouped

def compute_highway_speed_ci(df, sig_level=0.95, n_simulations=1000):
    # Group data by highway type
    grouped = df.groupby('highway')
    # Compute statistics for each group
    stats_list = []
    for name, group in grouped:
        speeds = group['maxspeed'].dropna().values
        if len(speeds) > 0:
            mean, lower, upper = compute_bootstrap_conf_intervals(speeds, sig_level, n_simulations)
            median = np.median(speeds)
            
            stats_list.append({
                'highway': name,
                'mean_speed': mean,
                'median': median,
                'lower_bound': lower,
                'upper_bound': upper,
                'sample_size': len(speeds)
            })
    # Convert results to DataFrame
    results = pd.DataFrame(stats_list)
    return results

def compute_lane_ci(df, sig_level=0.95, n_simulations=1000):
    # Group data by highway type
    grouped = df.groupby('highway')
    # Compute statistics for each group
    stats_list = []
    for name, group in grouped:
        speeds = group['lanes'].dropna().values
        if len(speeds) > 0:
            mean, lower, upper = compute_bootstrap_conf_intervals(speeds, sig_level, n_simulations)
            median = np.median(speeds)
            
            stats_list.append({
                'highway': name,
                'mean_lanes': mean,
                'median': median,
                'lower_bound': lower,
                'upper_bound': upper,
                'sample_size': len(speeds)
            })
    # Convert results to DataFrame
    results = pd.DataFrame(stats_list)
    return results

# fill in the missing rows with the confidence intervals
def fill_missing_values_row(row, column, median_dict):
    # column is either 'lanes' or 'maxspeed'
    if column == 'lanes':
        default_value = 2.0
    elif column == 'maxspeed':
        default_value = 20.0
    if pd.isna(row[column]):  # if lanes value is missing
        return median_dict.get(row['highway'], default_value)
    return row[column]

def fill_missing_values(edges_df, df_ci, column):
    # df_ci is either speed_ci or lane_ci
    # column is either 'lanes' or 'maxspeed'
    median_dict = dict(zip(df_ci['highway'], df_ci['median']))
    edges_df[column] = edges_df.apply(
        lambda row: fill_missing_values_row(row, column, median_dict), 
        axis=1
    )
    return edges_df

def fill_out_of_range_with_median(edges_df, column, range=None):
   # Create a copy to avoid modifying original
   result_df = edges_df.copy()
   # Calculate median for each highway type
   medians = edges_df.groupby('highway')[column].median()
   # Map medians to corresponding highway types
   highway_median = result_df['highway'].map(medians)
   # Create mask for None/NaN values
   null_mask = result_df[column].isna()
   if range is not None:
       # Create mask for out-of-range values and combine with null mask
       out_of_range_mask = (result_df[column] < (highway_median - range))
       combined_mask = out_of_range_mask | null_mask
   else:
       # Only use null mask if range not specified
       combined_mask = null_mask
   # Fill all masked values with their corresponding highway type median
   result_df.loc[combined_mask, column] = highway_median[combined_mask]
   return result_df


# -----------------
# (3) Prepare Highway Directions
# -----------------

# NOTE: this code below is unnecessary. All motorways are already one-way. I don't need to assign east west to them
# I don't need to rename nodes. I don't need to update the highway intersections

# this is complicated
# here's my problem: I have a list of edges and a list of nodes
# in my edge list, I can determine the directionality of one-way streets
# I can also assign directions to two-way streets easily
# Unfortunately, however, there's no assignment for side of the street
# If I'm on a highway, I might be at a node where there's an off-ramp, but I have no idea whether that off-ramp is on my side or not
# The data do not contain this information. If I worked with the data as is, I would be assuming that all off-ramps are available
# I think this assumption is fine for most local roads where there are 4 way stops, but it's not fine for highways
# Another example case is that these data allow for U-turns at any node. This is not realistic, especially for highways
# I can imagine a heavy traffic scenario where the model would assume people could U-turn out of highway traffic easily 

# Because I don't have data on side of the road, and because I'm building a macro model, I think it's fine to leave this assumption
# for the minor roads
# But for highways, I think this will have a meaningful impact on traffic flow. Basically, if there's a lot of morning traffic
# going into the city, the original data would assume double the number of off-ramps than there actually are. This would lead to
# much less predicted congestion than is actually the case. 
# I don't care about tagging the side of the off-ramp correctly (and I have no data to verify), 
# but I DO care about how I assign the sides along the highway
# i.e. this cannot be a random distribution; it must be even. Every other off-ramp should be on the opposite side of the highway. 
# This is how off-ramps are usually distributed

# what's difficult here, is that I need to come up with completely different node names for the highway nodes
# I'm going to use their original name + "_e" or "_w" for east and west (although the actual direction likely won't be east or west)
# then I need to flow this change throughout all the affected edges
# there are many edges that will touch a highway and will need to be update to reflect the new node names

# -----------------
# (3a) Read all edges as single direction 

def set_edge_category(row):
    ROAD_CATEGORIES = {
        'motorway': 'Major',
        'trunk': 'Major',
        'primary': 'Major',
        'secondary': 'Minor',
        'tertiary': 'Minor',
        'residential': 'Minor',
        'unclassified': 'Minor',
        'living_street': 'Minor',
        'motorway_link': 'Link',
        'trunk_link': 'Link',
        'primary_link': 'Link',
        'secondary_link': 'Link',
        'tertiary_link': 'Link'
    }
    return ROAD_CATEGORIES.get(row['highway'], 'Minor')  # Default to Minor if not found

# reversed is only true when oneway is false

def adjust_reverse_direction(edges_df):
    # (1) reset index
    edges_df = edges_df.reset_index(drop=True)
    # (2) swap Node1 and Node2 where reversed is True
    mask = edges_df['reversed'] == True
    edges_df.loc[mask, ['Node1', 'Node2']] = edges_df.loc[mask, ['Node2', 'Node1']].values
    # (3) drop reversed column
    edges_df = edges_df.drop('reversed', axis=1)
    return edges_df

# NOTE: I once thought this function was unnecessary, but it's actually very important
# at first it appeared that OpenStreetMap data already has both directions for 2-way streets -- with this function
# I was seeing so many duplicates
# but the problem is that OSM doesn't have both directions for all 2-way streets
# for example, edges_df[(edges_df['Node1'] == '110535017') & (edges_df['Node2'] == '8833967196')] has
# duplicate values for the same edge. i.e. there's no reverse edge, there's just two of the same edge even though it is listed as a 2-way street


# create a df of single directions that can be read Node1 -> Node2
def expand_2_way_edges(edges_df):
    # (1) reset index
    edges_df = edges_df.reset_index(drop=True)
    # (2) create duplicate rows for two-way edges
    two_way_edges = edges_df[edges_df['oneway'] == False].copy()
    two_way_edges[['Node1', 'Node2']] = two_way_edges[['Node2', 'Node1']]
    edges_df = pd.concat([edges_df, two_way_edges], ignore_index=True)
    # (3) drop oneway column
    # edges_df = edges_df.drop('oneway', axis=1).reset_index(drop=True)
    return edges_df

# -----------------
# (3b) Assign "_e" and "_w" side of road to highway nodes
# NOTE: this code below is unnecessary. All motorways are already one-way. I don't need to assign east west to them
# I don't need to rename nodes. I don't need to update the highway intersections

# steps
# filter for highway links only
# start with a highway link. Find the beginning of that highway (where the only links before it are non-highway links)
# run through that link until I get to the end of the highway (where the only links after it are non-highway links)
# assign a highway id and side of road tag (e.g. 'east', 'west') to those links
# find the corresponding opposite direction of that highway by finding links that are connected to the same nodes but in different directions
# assign the same highway id and the opposite side of road tag to those links
# continue to loop through until all highway links are assigned to a highway id and a side of road tag
# note: I'm determining 'highway' by connectivity rather than the 'name' which is a finicky label from the data

# Actually, I want to capture a list of beginning links and a list of ending links because highways can branch

# find all other links that are connected to the highway nodes
# I need to update those nodes to reflect the new side of road tag
# again, loop through the highway id's and alternate assignment _e and _w to these connected nodes
# need to make sure this works in cases where there's 2 on-ramps on the same node (one assigned _e and one assigned _w)

def drop_duplicates_except_lists(edges_df):
    # Define columns to check
    columns_to_check = [
        'Node1', 'Node2', 'key', 'osmid', 'highway', 'lanes', 
        'maxspeed', 'name', 'oneway'
    ]
    # Create a new DataFrame with just these columns
    temp_df = edges_df[columns_to_check].copy()
    # Convert all columns to strings to avoid any type issues
    for col in temp_df.columns:
        temp_df[col] = temp_df[col].astype(str)
    # Create a single string key for comparison
    temp_df['composite_key'] = temp_df.apply(lambda x: '_'.join(x), axis=1)
    # Get unique indices using the composite key
    unique_indices = temp_df.drop_duplicates('composite_key').index
    # Use these indices to filter the original dataframe
    edges_df = edges_df.loc[unique_indices].copy()
    edges_df = edges_df.reset_index(drop=True)
    return edges_df

# having an id for each link will help me with renaming the nodes
def tag_link_id(edges_df):
    print(f"number of rows before dropping duplicates: {edges_df.shape[0]}")
    # drop duplicate rows
    edges_df = drop_duplicates_except_lists(edges_df)
    print(f"number of rows after dropping duplicates: {edges_df.shape[0]}")
    edges_df = edges_df.reset_index(drop=True)
    edges_df['link_id'] = edges_df.index
    return edges_df

def filter_motorways(edges_df):
    motorways_df = edges_df[edges_df['highway'].isin(['motorway'])].copy()
    return motorways_df.reset_index(drop=True)

# Issue: I can't look at reverse links when trying to find the beginning of a highway. Don't want circularity to mess things up
# additionally, as I follow connectivity, the motorway can branch in complex ways. I need to record all the beginning links from all branches
# the 'ref' column is actually really helpful and robust. Hopefully it'll mean that I don't have to worry about branches
# the connectivity approach still matters because it handles edge cases and issues in the data

# # motorway_start_link_id can be a random 'link_id' from the motorways_df
# def find_beginning_of_motorway(motorway_start_link_id, motorways_df):
#    # (1) Initialize variables and output list
#    beginning_links = []
#    links_to_process = [motorway_start_link_id]
#    temp_df = motorways_df.copy()
#    start_link_ref = temp_df[temp_df['link_id'] == motorway_start_link_id].iloc[0]['ref']
#    print(f"start_link_ref: {start_link_ref}")
#    # (2) Continue processing until no more links to check
#    while links_to_process:
#        current_link_id = links_to_process.pop()
#        found_beginning = False
#        branch_temp_df = temp_df.copy()
#        branch_temp_df = branch_temp_df[branch_temp_df['ref'] == start_link_ref]
#        # (3) Follow current branch back to beginning
#        while not found_beginning:
#            # (4) Get current link's nodes
#            current_link = branch_temp_df[branch_temp_df['link_id'] == current_link_id].iloc[0]
#            current_node1 = current_link['Node1']
#            current_node2 = current_link['Node2']
#            # (5) Remove current link from temp_df so path can't circle back
#            branch_temp_df = branch_temp_df[branch_temp_df['link_id'] != current_link_id]
#            # (6) Remove reverse links so path can't circle back
#            branch_temp_df = branch_temp_df[~((branch_temp_df['Node1'] == current_node2) & 
#                                            (branch_temp_df['Node2'] == current_node1))]
#            # (7) Find links where Node2 matches current link's Node1  
#            previous_links = branch_temp_df[branch_temp_df['Node2'] == current_node1]
#            # (8) Check number of previous links
#            if len(previous_links) == 0:
#                # Found a beginning
#                beginning_links.append(current_link_id)
#                found_beginning = True
#            elif len(previous_links) > 1:
#                # Multiple branches - add to processing queue
#                for _, link in previous_links.iterrows():
#                    links_to_process.append(link['link_id'])
#                found_beginning = True
#            else:
#                # (9) Continue back one link
#                current_link_id = previous_links.iloc[0]['link_id']
#    # (10) Return list of beginning link_ids
#    return beginning_links

# def find_all_link_ids_for_single_motorway(motorway_start_link_id, motorways_df):
#     # (1) Initialize variables and output list
#     link_id_list = [motorway_start_link_id]
#     links_to_process = [motorway_start_link_id]
#     temp_df = motorways_df.copy()
#     start_link_ref = temp_df[temp_df['link_id'] == motorway_start_link_id].iloc[0]['ref']
#     branch_temp_df = temp_df[temp_df['ref'] == start_link_ref].copy()
    
#     # (2) Continue processing until no more links to check
#     while links_to_process:
#         current_link_id = links_to_process.pop()
#         current_link = branch_temp_df[branch_temp_df['link_id'] == current_link_id].iloc[0]
#         current_node2 = current_link['Node2']
        
#         # (3) Remove current link from temp_df
#         branch_temp_df = branch_temp_df[branch_temp_df['link_id'] != current_link_id]
        
#         # (4) Find only forward links where current Node2 matches next Node1
#         next_links = branch_temp_df[branch_temp_df['Node1'] == current_node2]
        
#         # (5) Add new forward links to processing queue and results list
#         for _, link in next_links.iterrows():
#             if link['link_id'] not in link_id_list:
#                 links_to_process.append(link['link_id'])
#                 link_id_list.append(link['link_id'])
    
#     return link_id_list

# # flip node 1 and node 2 to see if an opposite direction edge exists
# def find_opposite_direction_edge(motorway_link_id, motorways_df):
#     try:
#         # (1) Find the current link in the df
#         current_link = motorways_df[motorways_df['link_id'] == motorway_link_id].iloc[0]
#         current_node1 = current_link['Node1']
#         current_node2 = current_link['Node2']
#         # (2) Find the opposite direction
#         opposite_df = motorways_df[(motorways_df['Node1'] == current_node2) & 
#                                  (motorways_df['Node2'] == current_node1)]
#         if len(opposite_df) == 0:
#             return None
#         return opposite_df.iloc[0]['link_id']
#     except (IndexError, KeyError):
#         return None

# # I need to find the corresponding opposite direction highway so that I can tag it with the opposite side (_e, _w)
# # my hope is that all the highways are mirror opposites, but I can't guarantee that
# def find_opposite_direction_motorway_beginnings(original_motorway_link_id_list, motorways_df):
#     # (1) Initialize variables for loop
#     opposite_edges_to_process = []
#     beginning_links_list = []
#     all_links_list = [] # final list to return, also used to prevent redundant processing
#     # (2) Find opposite direction edges for all original links
#     for motorway_link_id in original_motorway_link_id_list:
#         try:
#             # (3) Find the opposite direction edge
#             opposite_edge = find_opposite_direction_edge(motorway_link_id, motorways_df)
#             if opposite_edge is not None:  # Only append if theres's a valid opposite edge
#                 opposite_edges_to_process.append(opposite_edge)
#         except (IndexError, KeyError):
#             # No opposite direction edge found
#             continue
#     # (4) Process each opposite direction edge
#     for opposite_edge in opposite_edges_to_process:
#         if opposite_edge not in all_links_list: # reduce redundant processing
#             # (5) Find beginning links for this opposite direction
#             opposite_beginnings_link_list = find_beginning_of_motorway(opposite_edge, motorways_df)
#             beginning_links_list.extend(opposite_beginnings_link_list)
#             # (6) Find all links for each beginning
#             for beginning_link in opposite_beginnings_link_list:
#                 if beginning_link not in all_links_list:
#                     all_links = find_all_link_ids_for_single_motorway(beginning_link, motorways_df)
#                     all_links_list.extend(all_links)
#     # (7) Remove duplicates while preserving order
#     beginning_links_list = list(dict.fromkeys(beginning_links_list))
#     all_links_list = list(dict.fromkeys(all_links_list))
#     return beginning_links_list, all_links_list

# -----------------
# (4) Prepare Free Flow Speed
# -----------------

# this is actually pretty complicated
# FFS isn't the same as max_speed of the road. There's acceleration and deacceleration due to stop lights, etc.
# In order to account for that, I'm creating a penalty variable that will be multiplied by the max_speed
# Later, I will calibrate this penalty variable with Google Maps API data on travel time
# This penalty is super important: it's how I will get my simulation to match real-world data

# For each edge, I am going to tag a category based on road type. The categories are Major, Minor, Link
# Then I look at Node2 of that edge and what other edges are connected. I create a matrix of these connections
# example: Major-Major, Major-Minor, Major-Link, etc.
# these 9 categories will be the different weights of the penalty variable
# by setting only 9 weights, I don't have to worry about overfitting
# Intuitively I believe in these 9 categories, going from a highway link to the same highway link, you probably don't slow down at all
# going from a minor road to a highway, there's probably a long light you need to wait for
# going from a major road to a minor road, you probably slow down a bit -- there's likely a stop sign, etc. 

# Note: I do have nodes data on traffic control from OpenStreet Map
# unfortunately these data are incomplete (they cover 13% of total nodes)
# while I wouldn't expect all nodes to have some kind of traffic marking, 13% feels too low

# Finally, I understand that there is a relationship between speed and FFS and also a relationship between edge length and FFS
# (1) Edge Length: if you slow to a stop on a 1 mile edge vs a 0.1 mile edge, your deacceleration distance is a much smaller fraction
# therefore, the adjustment to your max_speed to create FFS will be smaller. For a longer period of the road, you will be traveling at max speed
# (2) Speed: the faster you go, the more distance it takes to deaccelerate. Therefore, the impact on your FFS will be greater
# on point (2), speed is also correlated with a lower penalty. On faster roads, there are fewer lights or at the very least, lights are in sync
# I am choosing not to represent these relationships in this analysis because I'm not sure what the exact relationship is
# is it linear? quadratic? something more complex?
# obviously speed is super complex, but for the most part, road type correlates with speed
# I care about macro scale interventions, so I think it's fine to ignore these relationships
# additionally, I don't want to add unnecessary complexity. Right now, the relationship between the penalty weight and FFS is very clear
# I can understand and justify that relationship. It helps when I set initial values to the penalty weights

# this makes things more complicated than creating a df of unique Node2's because the intersections 

# create a column that is filled with a list of the connected edges
# def identify_node2_intersections(edges_df):
#     # (1) Initialize variables
#     edges_df['connected_edges'] = None
#     # (2) Loop through each edge
#     for index, row in edges_df.iterrows():
#         edge_node1 = row['Node1']
#         edge_node2 = row['Node2']
#         # make sure to drop the reversed Node1 Node2
#         # so if I'm looking at an edge that's a 2-way street, I don't want to count the edge that's going the opposite direction
#         edges_df_drop_reversed = edges_df[(edges_df['Node1'] != edge_node2) & (edges_df['Node2'] != edge_node1)]
#         # (3) Find all edges connected to Node2
#         connected_edges = edges_df_drop_reversed[edges_df_drop_reversed['Node1'] == edge_node2]['link_id'].tolist()
#         # (4) Remove current edge from list
#         connected_edges = [x for x in connected_edges if x != row['link_id']]
#         # (5) Assign list to current edge
#         edges_df.at[index, 'connected_edges'] = connected_edges
#     return edges_df

# this version was wrong. it wholesale removed all reverse directions from the comparison
# I only want to remove the reverse for that specific row 
# def identify_node2_intersections(edges_df):
#     # (1) Create dictionaries for faster lookup in both directions
#     node1_to_links = edges_df.groupby('Node1')['link_id'].apply(list).to_dict()
#     node2_to_links = edges_df.groupby('Node2')['link_id'].apply(list).to_dict()
#     edge_pairs = set(zip(edges_df['Node1'], edges_df['Node2']))
#     link_ids = edges_df['link_id'].tolist()
#     # (2) Initialize results list
#     results = []
#     # (3) Loop through each edge once
#     for index, row in edges_df.iterrows():
#         edge_node1 = row['Node1']
#         edge_node2 = row['Node2']
#         current_link = row['link_id']
#         # (4) Get connected edges from both Node1 and Node2 connections
#         connected_edges_from_node2 = node1_to_links.get(edge_node2, [])
#         connected_edges_to_node2 = node2_to_links.get(edge_node2, [])
#         # (5) Combine and filter out reversed edges and current edge
#         all_connected = connected_edges_from_node2 + connected_edges_to_node2
#         filtered_edges = [link for link in all_connected 
#                          if link != current_link 
#                          and (edge_node2, edge_node1) not in edge_pairs]
#         # (6) Remove duplicates while maintaining list format
#         filtered_edges = list(dict.fromkeys(filtered_edges))
#         results.append(filtered_edges)
#     # (7) Bulk assign results
#     edges_df['connected_edges'] = results
#     return edges_df

def identify_node2_intersections(edges_df):
    # Create dictionaries for faster lookup in both directions
    node1_to_links = edges_df.groupby('Node1')['link_id'].apply(list).to_dict()
    node2_to_links = edges_df.groupby('Node2')['link_id'].apply(list).to_dict()
    
    # Create a dictionary to look up reverse edges by link_id
    edge_lookup = edges_df.set_index('link_id')[['Node1', 'Node2']].to_dict('index')
    
    results = []
    
    for index, row in edges_df.iterrows():
        edge_node1 = row['Node1']
        edge_node2 = row['Node2']
        current_link = row['link_id']
        
        # Get connected edges from both Node1 and Node2 connections
        connected_edges_from_node2 = node1_to_links.get(edge_node2, [])
        connected_edges_to_node2 = node2_to_links.get(edge_node2, [])
        
        # Combine and filter
        all_connected = connected_edges_from_node2 + connected_edges_to_node2
        filtered_edges = []
        
        for link in all_connected:
            if link == current_link:
                continue
                
            # Only exclude if this specific link is the exact reverse of current edge
            link_nodes = edge_lookup.get(link)
            if link_nodes and link_nodes['Node1'] == edge_node2 and link_nodes['Node2'] == edge_node1:
                continue
                
            filtered_edges.append(link)
        
        # Remove duplicates while maintaining list format
        filtered_edges = list(dict.fromkeys(filtered_edges))
        results.append(filtered_edges)
    
    edges_df['connected_edges'] = results
    return edges_df

def identify_node2_edge_categories(edges_df):
    #(1) Iterate through each row in the edges dataframe
    for index, row in edges_df.iterrows():
        #(2) Get list of connected edges from the current row
        connected_edge_list = row['connected_edges']
        #(3) Find unique categories of connected edges
        connected_categories = edges_df[edges_df['link_id'].isin(connected_edge_list)]['edge_category'].unique().tolist()
        #(4) Assign the list of connected categories to the new column
        edges_df.at[index, 'connected_edge_categories'] = connected_categories
    #(5) Return the modified dataframe
    return edges_df

def identify_single_connected_edge_category(edges_df):
    #(1) Define priority order for categories
    priority_order = ['Major', 'Link', 'Minor']
    #(2) Create new column to store single category
    edges_df['single_connected_category'] = None
    #(3) Iterate through each row
    for index, row in edges_df.iterrows():
        #(4) Get list of connected categories for current row
        categories = row['connected_edge_categories']
        #(5) Find highest priority category that exists in the list
        for category in priority_order:
            if category in categories:
                edges_df.at[index, 'single_connected_category'] = category
                break
    edges_df['single_connected_category'] = edges_df['single_connected_category'].fillna('no_connections')
    #(6) Return modified dataframe
    return edges_df

def add_category_matrix_column(edges_df):
    edges_df['category_matrix'] = edges_df['edge_category'] + '-' + edges_df['single_connected_category']
    return edges_df

# create link penalty weights based on the category matrix

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

# -----------------
# calculate the other variables necessary for the analysis

# Jam density is the maximum number of vehicles that can physically fit on a road segment when traffic is completely stopped (jammed)
# Unit: number of vehicles per link/edge
def calculate_jam_density(edges_df):
    avg_vehicle_length = 4.5  # in meters
    avg_vehicle_spacing = 2  # in meters
    # note: length and lanes are never null in the OpenStreetMaps data
    edges_df['jam_density_kj'] = (edges_df['lanes'] * edges_df['length']) / (avg_vehicle_length + avg_vehicle_spacing)
    return edges_df

# percent heavy vehicles affects the capacity estimate
# note: in Oahu, I'm imagining there are very few heavy vehicles because it's an island
def calculate_pct_heavy_vehicle(edges_df):
    # Very low percentages reflecting Oahu's minimal heavy vehicle traffic
    hv_percentages = {
        'motorway': 3,      # H1/H2/H3 - Highest near port/airport
        'motorway_link': 2, 
        'trunk': 2,         # Major arterials
        'trunk_link': 2,
        'primary': 1,       # Major streets
        'primary_link': 1,
        'secondary': 0.5,   # Local collectors
        'secondary_link': 0.5,
        'tertiary': 0.5,    # Local streets
        'tertiary_link': 0.5,
        'residential': 0.1, # Residential streets
        'living_street': 0.1,
        'unclassified': 0.1
    }
    pct_HV = edges_df['highway'].map(hv_percentages)
    return pct_HV

# capacity is measured in vehicles per hour
def calculate_capacity(edges_df):
    numerator = 2200 + (10 * (np.minimum(70, edges_df['FFS']) - 50))  # vehicles per hour per lane
    denominator = 1 + (edges_df['pct_HV'] / 100)  # dimensionless adjustment for heavy vehicles
    edges_df['capacity'] = (numerator / denominator) * edges_df['lanes']  # vehicles per hour for entire link
    return edges_df

# I'm going to set a fixed wavespeed instead of calculating it as a unique value for each edge
# I'm going with -12mph

# Critical density is the density at which traffic flow reaches its maximum flow (capacity). It's a sweet spot
# Below k_c: traffic flows freely, adding more vehicles increases flow
# Above k_c: traffic becomes congested, adding more vehicles reduces flow
# At k_c: you achieve maximum flow (capacity)
# Units: vehicles per mile
def calculate_critical_density(edges_df):
    edges_df['critical_density_kc'] = edges_df['capacity'] / edges_df['FFS']
    return edges_df

# Queue discharge rate: how quickly vehicles can exit a queue once congestion starts clearing
# 0.95% is in-line with other studies
def calculate_queue_discharge_rate(edges_df):
    edges_df['queue_discharge_rate_q'] = 0.95 * edges_df['capacity']
    return edges_df

# Unit: minutes
def calculate_free_flow_travel_time(edges_df):
    # FFS is in miles per hour so I need to convert it to miles per minute
    # length is in meters so I need to convert it to miles
    edges_df['length_miles'] = edges_df['length'] / 1609.34
    edges_df['free_flow_travel_time'] = (edges_df['length_miles']) / (edges_df['FFS'] / 60)
    return edges_df

# -----------------
# (5) Main functions
# -----------------

def read_nodes_and_edges(G):
    nodes_df = pd.DataFrame(G.nodes(data=True), columns=['Node', 'Attributes'])
    edges_df = pd.DataFrame(G.edges(data=True), columns=['Node1', 'Node2', 'Attributes'])
    return nodes_df, edges_df

def prepare_edges_basic(G):
    edges_df = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
    edges_df = edges_df.rename(columns={'u': 'Node1', 'v': 'Node2'})
    edges_df['Node1'] = edges_df['Node1'].astype(str)
    edges_df['Node2'] = edges_df['Node2'].astype(str)
    # fix highway
    edges_df['highway'] = edges_df['highway'].apply(categorize_highway)
    edges_df = edges_df[edges_df['highway'] != 'escape']
    # fix max speed
    edges_df['maxspeed'] = edges_df['maxspeed'].apply(clean_max_speed)
    # fix lanes
    edges_df['lanes'] = edges_df['lanes'].apply(clean_lanes)
    # understand edges
    understand_edges(edges_df)
    return edges_df

def prepare_edges_ci(edges_df):

    # I included corrections for out of range
    # I only use median, so there's no need in calculating ci anymore
    # In fact, ci is less informative than calculating the min and max to see if there are erroneous edges
    edges_df = fill_out_of_range_with_median(edges_df, 'maxspeed', range=5)
    edges_df = fill_out_of_range_with_median(edges_df, 'lanes')

    # # calculate the ci's
    # speed_ci = compute_highway_speed_ci(edges_df)
    # lane_ci = compute_lane_ci(edges_df)
    # # fill in missing values with the median from the confidence intervals
    # edges_df = fill_missing_values(edges_df, speed_ci, 'maxspeed')
    # edges_df = fill_missing_values(edges_df, lane_ci, 'lanes')
    return edges_df

# def filter_edges(edges_df, highway_list):
#     # highway_list = ['residential', 'living_street', 'unclassified']
#     print(f"number of edges before filtering: {edges_df.shape[0]}")
#     edges_df = edges_df[~edges_df['highway'].isin(highway_list)].copy()
#     print(f"number of edges after filtering: {edges_df.shape[0]}")
#     return edges_df

def prepare_edges_matrix(edges_df):
    edges_df['edge_category'] = edges_df.apply(set_edge_category, axis=1)
    edges_df = adjust_reverse_direction(edges_df) # for one-way streets
    # adding in reverse direction for two-way streets now 
    edges_df = expand_2_way_edges(edges_df)
    # I wonder if these duplicate values are coming from the number of lanes
    edges_df = tag_link_id(edges_df) # includes a drop duplicate rows
    edges_df = identify_node2_intersections(edges_df)
    edges_df = identify_node2_edge_categories(edges_df)
    edges_df = identify_single_connected_edge_category(edges_df)
    edges_df = add_category_matrix_column(edges_df)
    return edges_df

def create_network_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
    # Initialize empty directed graph
    G = nx.DiGraph()
    # Add edges with all their attributes
    for _, row in edges_df.iterrows():
        G.add_edge(
            row['Node1'],                    # From node
            row['Node2'])                    # To node
    # Basic validation
    print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    # Check for disconnected components
    if not nx.is_strongly_connected(G):
        print("Warning: Network contains disconnected components")
        components = list(nx.strongly_connected_components(G))
        print(f"Number of strongly connected components: {len(components)}")
    return G

# def filter_edges(edges_df, highway_list):
#     # First pass: keep all non-local roads
#     main_roads = edges_df[~edges_df['highway'].isin(highway_list)].copy()
    
#     # Create graph of main roads
#     G_main = create_network_graph(main_roads)
#     components_before = list(nx.strongly_connected_components(G_main))
    
#     # Second pass: identify local roads that connect different components
#     connecting_roads = []
#     for _, row in edges_df[edges_df['highway'].isin(highway_list)].iterrows():
#         node1, node2 = row['Node1'], row['Node2']
#         # Check if this local road connects different components
#         for comp in components_before:
#             if (node1 in comp) != (node2 in comp):  # One node in, one node out
#                 connecting_roads.append(row['link_id'])
#                 break
    
#     # Keep main roads plus connecting local roads
#     final_edges = pd.concat([
#         main_roads,
#         edges_df[edges_df['link_id'].isin(connecting_roads)]
#     ])
    
#     print(f"Original edges: {len(edges_df)}")
#     print(f"After filtering: {len(final_edges)}")
#     print(f"Kept {len(connecting_roads)} connecting local roads")
    
#     return final_edges

# def filter_largest_component(edges_df: pd.DataFrame) -> pd.DataFrame:
#     """Returns only the edges that form the largest strongly connected component"""
#     #1. Create initial network graph
#     G = create_network_graph(edges_df)
#     #2. Find all strongly connected components
#     components = list(nx.strongly_connected_components(G))
#     #3. Get the largest component
#     largest_component = max(components, key=len)
#     print(f"Largest component has {len(largest_component)} nodes")
#     #4. Filter edges to only those where both nodes are in largest component
#     mask = edges_df.apply(lambda row: row['Node1'] in largest_component and row['Node2'] in largest_component, axis=1)
#     filtered_edges = edges_df[mask].copy()
#     #5. Validate final network
#     G_final = create_network_graph(filtered_edges)
#     print(f"Final network has {len(filtered_edges)} edges")
#     if not nx.is_strongly_connected(G_final):
#         print("Error: Final network is not strongly connected")
#     return filtered_edges

# this function is for strongly_connected_components which are bidirectional
def filter_top_components(edges_df: pd.DataFrame, n_components: int = 4) -> pd.DataFrame:
   """Returns edges from the top N largest strongly connected components with component IDs"""
   #1. Create initial network graph
   G = create_network_graph(edges_df)
   #2. Find all strongly connected components
   components = list(nx.strongly_connected_components(G))
   #3. Sort components by size and get top N
   top_components = sorted(components, key=len, reverse=True)[:n_components]
   #4. Create mapping of nodes to component IDs
   node_to_component = {}
   for i, component in enumerate(top_components):
       for node in component:
           node_to_component[node] = i
       print(f"Component {i} has {len(component)} nodes")
   #5. Filter edges and assign component IDs
   mask = edges_df.apply(lambda row: (row['Node1'] in node_to_component) and 
                                   (row['Node2'] in node_to_component), axis=1)
   filtered_edges = edges_df[mask].copy()
   #6. Add component_id column
   filtered_edges['component_id'] = filtered_edges.apply(
       lambda row: node_to_component[row['Node1']], axis=1)
   #7. Validate final components
   print(f"\nFinal network has {len(filtered_edges)} edges")
   print(f"Component distribution:")
   print(filtered_edges['component_id'].value_counts())
   return filtered_edges

# this function is for connected_components which are unidirectional
# When I run `strongly_connected_components()`, it's looking for paths where I can get 
# from any node to any other node in BOTH directions. But if I have:
# - A highway going into town on one set of nodes
# - A highway going out of town on a completely different set of nodes
# - These node sets never "connect" to each other

# def filter_top_components(edges_df: pd.DataFrame, n_components: int = 4) -> pd.DataFrame:
#     """Returns edges from the top N largest connected components with component IDs"""
#     #1. Create initial network graph - explicitly make it undirected
#     G = create_network_graph(edges_df).to_undirected()

#     #2. Find all connected components (undirected)
#     components = list(nx.connected_components(G))
    
#     #3. Sort components by size and get top N
#     top_components = sorted(components, key=len, reverse=True)[:n_components]
    
#     #4. Create mapping of nodes to component IDs
#     node_to_component = {}
#     for i, component in enumerate(top_components):
#         for node in component:
#             node_to_component[node] = i
#         print(f"Component {i} has {len(component)} nodes")
    
#     #5. Filter edges and assign component IDs
#     mask = edges_df.apply(lambda row: (row['Node1'] in node_to_component) and 
#                          (row['Node2'] in node_to_component), axis=1)
#     filtered_edges = edges_df[mask].copy()
    
#     #6. Add component_id column
#     filtered_edges['component_id'] = filtered_edges.apply(
#         lambda row: node_to_component[row['Node1']], axis=1)
    
#     #7. Validate final components
#     print(f"\nFinal network has {len(filtered_edges)} edges")
#     print(f"Component distribution:")
#     print(filtered_edges['component_id'].value_counts())
    
#     return filtered_edges

def prepare_nodes_basic(G, edges_df):
    nodes_df = ox.graph_to_gdfs(G, nodes=True, edges=False).reset_index()
    # 'Nodes' is incorrectly being renamed to osmid
    nodes_df = nodes_df.rename(columns={'osmid': 'Node'})
    nodes_df['Node'] = nodes_df['Node'].astype(str)

    # filter nodes_df for only the nodes that are in the edges_df
    unique_nodes = pd.concat([edges_df['Node1'], edges_df['Node2']]).unique()
    print(f"number of nodes before filtering: {nodes_df.shape[0]}")
    nodes_df = nodes_df[nodes_df['Node'].isin(unique_nodes)].copy()
    print(f"number of nodes after filtering: {nodes_df.shape[0]}")
    return nodes_df

def calculate_highway_variables(edges_df, penalty_dictionary):
    edges_df = create_penalty_weights(edges_df, penalty_dictionary)
    edges_df['FFS'] = calculate_FFS(edges_df)
    edges_df['pct_HV'] = calculate_pct_heavy_vehicle(edges_df)
    edges_df = calculate_jam_density(edges_df)
    edges_df = calculate_capacity(edges_df)
    edges_df = calculate_critical_density(edges_df)
    edges_df = calculate_queue_discharge_rate(edges_df)
    edges_df = calculate_free_flow_travel_time(edges_df)
    return edges_df

# I don't want to filter highway_list because I don't want to accidentally lose strong components
# highway_list = ['residential', 'living_street', 'unclassified']
def main_preparation(penalty_dictionary, place_query = 'Oahu, Hawaii, USA'):
    G = ox.graph_from_place(place_query, network_type='drive')
    # nodes_df, edges_df = read_nodes_and_edges(G)
    edges_df = prepare_edges_basic(G)
    edges_df = prepare_edges_ci(edges_df)
    # edges_df = filter_edges(edges_df, highway_list)
    # nodes_df = prepare_nodes_basic(G, edges_df)
    edges_df = prepare_edges_matrix(edges_df)
    # edges_df = filter_edges(edges_df, highway_list) # moving filter_edges() after prepare_edges_matrix()
    edges_df = filter_top_components(edges_df)

    # I'm adding code here to just take the largest strongly connected component
    edges_df = edges_df[edges_df['component_id'] == 0].copy()

    nodes_df = prepare_nodes_basic(G, edges_df) # nodes preparation must come after filtering edges
    edges_df = calculate_highway_variables(edges_df, penalty_dictionary)
    return nodes_df, edges_df
