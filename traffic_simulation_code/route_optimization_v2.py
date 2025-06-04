# the purpose of this to use the edges_df and od_matrix to optimize the routes

# import necessary packages
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple
import os
import geopandas as gpd


# -----------------
# (1) Initial graph creation and optimization
# -----------------

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

def calculate_initial_routes(G: nx.DiGraph, od_matrix: pd.DataFrame, edges_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, List[int]], pd.DataFrame]:
    #(1) Verify all OD nodes are truly on two-way streets
    print("\nVerifying OD matrix nodes:")
    all_od_nodes = pd.concat([od_matrix['origin_node'], od_matrix['destination_node']]).unique()
    print(f"Total unique nodes in OD matrix: {len(all_od_nodes)}")
    for node in all_od_nodes:
        node = str(node)  # ensure string type
        # Count occurrences as Node1 and Node2
        as_node1 = edges_df[edges_df['Node1'] == node].shape[0]
        as_node2 = edges_df[edges_df['Node2'] == node].shape[0]
        if as_node1 == 0 or as_node2 == 0:
            print(f"Node {node} appears as Node1: {as_node1} times, as Node2: {as_node2} times")
            # Show the actual edges for this node
            print("Edges:")
            print(edges_df[edges_df['Node1'] == node][['Node1', 'Node2', 'oneway', 'link_id']])
            print(edges_df[edges_df['Node2'] == node][['Node1', 'Node2', 'oneway', 'link_id']])
            break
    #(2) Create node pair to link_id mapping
    link_lookup = {(str(row['Node1']), str(row['Node2'])): row['link_id'] for _, row in edges_df.iterrows()}
    #(3) Initialize output dataframe
    vehicles_df = od_matrix.copy()
    vehicles_df['current_link'] = None
    vehicles_df['route_links'] = None
    vehicles_df['routing_status'] = 'pending'
    #(4) Calculate shortest paths for unique OD pairs
    route_dict = {}
    unique_od_pairs = od_matrix.groupby(['origin_node', 'destination_node']).size().reset_index()
    print(f"Calculating routes for {len(unique_od_pairs)} unique OD pairs...")
    #(5) Track statistics and failures
    successful_routes = 0
    failed_routes = 0
    failures_list = []
    unreachable_pairs = set()
    for _, row in unique_od_pairs.iterrows():
        origin = str(row['origin_node'])
        destination = str(row['destination_node'])
        if (origin, destination) in unreachable_pairs:
            continue
        try:
            if origin not in G or destination not in G:
                failures_list.append({
                    'origin': origin,
                    'destination': destination,
                    'failure_type': 'node_not_in_network',
                    'origin_in_network': origin in G,
                    'destination_in_network': destination in G
                })
                unreachable_pairs.add((origin, destination))
                continue
            path_nodes = nx.shortest_path(G, origin, destination, weight='weight')
            #(6) Convert to link IDs
            path_links = []
            for i in range(len(path_nodes) - 1):
                node_pair = (path_nodes[i], path_nodes[i + 1])
                if node_pair not in link_lookup:
                    print(f"Warning: Missing link between nodes {node_pair}")
                    continue
                link_id = link_lookup[node_pair]
                path_links.append(link_id)
            #(7) Only process if we found a valid path
            if path_links:
                mask = (od_matrix['origin_node'] == row['origin_node']) & (od_matrix['destination_node'] == row['destination_node'])
                person_ids = od_matrix.loc[mask, 'person_id']
                for pid in person_ids:
                    route_dict[pid] = path_links
                    vehicles_df.loc[vehicles_df['person_id'] == pid, 'route_links'] = str(path_links)
                    vehicles_df.loc[vehicles_df['person_id'] == pid, 'current_link'] = path_links[0]
                    vehicles_df.loc[vehicles_df['person_id'] == pid, 'routing_status'] = 'success'
                    successful_routes += 1
            else:
                failures_list.append({
                    'origin': origin,
                    'destination': destination,
                    'failure_type': 'no_valid_links'
                })
                unreachable_pairs.add((origin, destination))
                failed_routes += 1
        except nx.NetworkXNoPath:
            failures_list.append({
                'origin': origin,
                'destination': destination,
                'failure_type': 'no_path_found'
            })
            unreachable_pairs.add((origin, destination))
            failed_routes += 1
            continue
    #(8) Create failures DataFrame
    failures_df = pd.DataFrame(failures_list)
    #(9) Print enhanced summary statistics
    print("\nRouting Summary:")
    print(f"Successfully routed: {successful_routes} vehicles")
    print(f"Failed to route: {failed_routes} OD pairs")
    print(f"Total unreachable OD pairs: {len(unreachable_pairs)}")
    if not failures_df.empty:
        print("\nFailure Analysis:")
        print(f"Number of unique failing origin nodes: {failures_df['origin'].nunique()}")
        print(f"Number of unique failing destination nodes: {failures_df['destination'].nunique()}")
        print("\nTop 5 problematic origin nodes:")
        origin_counts = failures_df['origin'].value_counts().head()
        print(origin_counts)
        print("\nTop 5 problematic destination nodes:")
        dest_counts = failures_df['destination'].value_counts().head()
        print(dest_counts)
    #(10) Mark vehicles with no route
    mask = vehicles_df['routing_status'] == 'pending'
    vehicles_df.loc[mask, 'routing_status'] = 'failed'
    unrouted = vehicles_df['route_links'].isna().sum()
    if unrouted > 0:
        print(f"Total vehicles without routes: {unrouted}")
    return vehicles_df, failures_df

# -----------------
# (2) Initialize traffic state and vehicle state
# -----------------

def initialize_traffic_state(edges_df: pd.DataFrame) -> pd.DataFrame:
    #(1) make a copy to avoid modifying original
    dynamic_edges_df = edges_df.copy()
    #(2) add columns for tracking current traffic state
    dynamic_edges_df['current_k'] = 0.0  # density (vehicles/mile)
    dynamic_edges_df['current_q'] = 0.0  # flow rate (vehicles/hour)
    dynamic_edges_df['queue_length'] = 0  # number of vehicles queued
    dynamic_edges_df['vehicles_on_link'] = [[] for _ in range(len(edges_df))]  # list of vehicle ids currently on link
    #(3) add columns for performance metrics
    dynamic_edges_df['current_speed'] = dynamic_edges_df['FFS']  # initialize at free flow speed
    dynamic_edges_df['current_travel_time'] = dynamic_edges_df['free_flow_travel_time']
    #(4) verify all required columns exist
    required_cols = ['capacity', 'length_miles', 'FFS', 'jam_density_kj']
    missing_cols = [col for col in required_cols if col not in dynamic_edges_df.columns]
    if missing_cols:
        raise ValueError(f"missing required columns: {missing_cols}")
    return dynamic_edges_df

def initialize_vehicle_state(vehicles_df: pd.DataFrame) -> pd.DataFrame:
    #(1) make a copy to avoid modifying original
    dynamic_vehicles_df = vehicles_df.copy()
    #(2) add columns for tracking vehicle state
    dynamic_vehicles_df['position_on_link_meters'] = 0.0  # meters from start of current link
    dynamic_vehicles_df['accumulated_travel_time'] = 0.0  # total travel time so far
    dynamic_vehicles_df['status'] = 'waiting'  # waiting, active, queued, or finished
    dynamic_vehicles_df['next_link_index'] = 0  # index into route_links list
    #(3) convert route_links from string to actual list
    dynamic_vehicles_df['route_links'] = dynamic_vehicles_df['route_links'].apply(eval)
    #(4) verify all vehicles have valid routes
    missing_routes = dynamic_vehicles_df[dynamic_vehicles_df['routing_status'] != 'success']
    if not missing_routes.empty:
        print(f"warning: {len(missing_routes)} vehicles have no valid routes")
    #(5) add validation columns
    dynamic_vehicles_df['route_progress'] = 0.0  # percentage of route completed
    dynamic_vehicles_df['expected_arrival_time'] = None  # estimated time of arrival
    return dynamic_vehicles_df

# -----------------
# (3) Set up vehicle movement and flow dynamics
# -----------------

# def calculate_link_conditions(edges_df: pd.DataFrame, vehicles_df: pd.DataFrame) -> pd.DataFrame:
#     #(1) make a copy to avoid modifying original
#     updated_edges_df = edges_df.copy()
#     #(2) process each link
#     w = -12  # wave speed in mph
#     for link_id, link_data in updated_edges_df.iterrows():
#         #(3) calculate density (vehicles/mile)
#         vehicles_on_link = len(vehicles_df[vehicles_df['current_link'] == link_id])
#         # current_k should be in vehicles per mile because that is the unit for critical_density_kc
#         current_k = vehicles_on_link / link_data['length_miles']
#         updated_edges_df.at[link_id, 'current_k'] = current_k
#         #(4) calculate flow based on traffic state
#         if current_k <= link_data['critical_density_kc']:
#             current_q = link_data['FFS'] * current_k
#         else:
#             current_q = w * (current_k - link_data['jam_density_kj'])
#         updated_edges_df.at[link_id, 'current_q'] = current_q
#         #(5) calculate queue length
#         prev_queue = link_data['queue_length']
#         # note: need to implement vehicles arriving/departing count
#         arriving = 0  # placeholder
#         departing = 0  # placeholder
#         new_queue = prev_queue + arriving - departing
#         #(6) check for spillback
#         max_queue = (link_data['length_miles'] * link_data['lanes'] * 
#                     link_data['jam_density_kj']) - vehicles_on_link
#         new_queue = min(new_queue, max_queue)
#         updated_edges_df.at[link_id, 'queue_length'] = new_queue
#     return updated_edges_df

def calculate_link_conditions(edges_df: pd.DataFrame, vehicles_df: pd.DataFrame) -> pd.DataFrame:
    #(1) make a copy to avoid modifying original
    updated_edges_df = edges_df.copy()
    #(2) process each link
    w = -12  # wave speed in mph
    for link_id, link_data in updated_edges_df.iterrows():
        #(3) calculate density (vehicles/mile)
        vehicles_on_link = len(vehicles_df[vehicles_df['current_link'] == link_id])
        current_k = vehicles_on_link / link_data['length_miles']
        updated_edges_df.at[link_id, 'current_k'] = current_k
        #(4) calculate flow based on traffic state
        if current_k <= link_data['critical_density_kc']:
            current_q = link_data['FFS'] * current_k
        else:
            current_q = w * (current_k - link_data['jam_density_kj'])
        updated_edges_df.at[link_id, 'current_q'] = current_q
        #(5) calculate arriving and departing vehicles
        active_vehicles = vehicles_df[vehicles_df['status'] == 'active']
        route_links = active_vehicles['route_links'].to_list()
        next_link_indices = active_vehicles['next_link_index'].to_list()
        current_links = active_vehicles['current_link'].to_list()
        positions = active_vehicles['position_on_link_meters'].to_list()
        #(6) count arriving vehicles (those about to enter this link)
        arriving = sum(1 for i in range(len(route_links))
                      if (isinstance(route_links[i], list) and 
                          next_link_indices[i] < len(route_links[i]) and
                          route_links[i][next_link_indices[i]] == link_id))
        #(7) count departing vehicles (those ready to leave this link)
        departing = sum(1 for i in range(len(route_links))
                       if (current_links[i] == link_id and
                           positions[i] >= link_data['length']))
        #(8) calculate queue length
        prev_queue = link_data['queue_length']
        new_queue = prev_queue + arriving - departing
        #(9) check for spillback
        max_queue = (link_data['length_miles'] * link_data['lanes'] * link_data['jam_density_kj']) - vehicles_on_link
        new_queue = min(new_queue, max_queue)
        updated_edges_df.at[link_id, 'queue_length'] = max(0, new_queue)
    return updated_edges_df

def update_vehicle_positions(vehicles_df: pd.DataFrame, edges_df: pd.DataFrame, 
                           timestep: float) -> pd.DataFrame:
    #(1) make a copy to avoid modifying original
    updated_vehicles_df = vehicles_df.copy()
    #(2) update only active vehicles
    active_vehicles = updated_vehicles_df[updated_vehicles_df['status'] == 'active']
    for idx, vehicle in active_vehicles.iterrows():
        current_link = vehicle['current_link']
        link_data = edges_df.loc[current_link]
        #(3) update position
        if link_data['current_k'] <= link_data['critical_density_kc']:
            # free flow conditions
            speed = link_data['FFS']  # mph
            distance = speed * timestep  # miles
            distance_meters = distance * 1609.34  # convert miles to meters
            new_position = vehicle['position_on_link_meters'] + distance_meters
        else:
            # congested conditions, handle queueing
            if link_data['queue_length'] > 0:
                # vehicle is in queue, limited movement
                speed = link_data['queue_discharge_rate_q'] / link_data['current_k']
                distance = speed * timestep
                distance_meters = distance * 1609.34  # convert miles to meters
                new_position = vehicle['position_on_link_meters'] + distance_meters
            else:
                speed = link_data['FFS'] * (1 - link_data['current_k'] / 
                                          link_data['jam_density_kj'])
                distance = speed * timestep
                distance_meters = distance * 1609.34  # convert miles to meters
                new_position = vehicle['position_on_link_meters'] + distance_meters
        #(4) check for link transition
        if new_position >= link_data['length']:
            route_links = vehicle['route_links']
            next_link_idx = vehicle['next_link_index'] + 1
            if next_link_idx >= len(route_links):
                updated_vehicles_df.at[idx, 'status'] = 'finished'
            else:
                updated_vehicles_df.at[idx, 'current_link'] = route_links[next_link_idx]
                updated_vehicles_df.at[idx, 'next_link_index'] = next_link_idx
                updated_vehicles_df.at[idx, 'position_on_link_meters'] = 0.0
        else:
            updated_vehicles_df.at[idx, 'position_on_link_meters'] = new_position
        #(5) update travel time
        # timestep is in hours
        # accumulated_travel_time is in minutes
        updated_vehicles_df.at[idx, 'accumulated_travel_time'] += timestep * 60  # convert hours to minutes
    return updated_vehicles_df

# -----------------
# (4) Queue Management
# -----------------

def manage_queues(edges_df: pd.DataFrame, vehicles_df: pd.DataFrame) -> pd.DataFrame:
    #(1) make a copy to avoid modifying original
    updated_edges_df = edges_df.copy()
    #(2) process each link
    for link_id, link_data in updated_edges_df.iterrows():
        #(3) calculate current link utilization
        current_vehicles = len(vehicles_df[vehicles_df['current_link'] == link_id])
        max_vehicles = link_data['length_miles'] * link_data['lanes'] * link_data['jam_density_kj']
        #(4) calculate available capacity
        available_space = max_vehicles - current_vehicles
        #(5) update queue if at capacity
        if available_space <= 0:
            updated_edges_df.at[link_id, 'queue_length'] += 1
        #(6) process queue discharge
        elif link_data['queue_length'] > 0:
            discharge_rate = min(link_data['queue_discharge_rate_q'], available_space)
            updated_edges_df.at[link_id, 'queue_length'] = max(0, link_data['queue_length'] - discharge_rate)
    return updated_edges_df

def process_link_transitions(vehicles_df: pd.DataFrame, edges_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #(1) make copies to avoid modifying originals
    updated_vehicles_df = vehicles_df.copy()
    updated_edges_df = edges_df.copy()
    #(2) find vehicles at link boundaries
    transitioning_vehicles = updated_vehicles_df[
        (updated_vehicles_df['status'] == 'active') & 
        (updated_vehicles_df['position_on_link_meters'] >= edges_df.loc[updated_vehicles_df['current_link'], 'length'].values)
    ]
    #(3) process each transitioning vehicle
    for idx, vehicle in transitioning_vehicles.iterrows():
        current_link = vehicle['current_link']
        route_links = vehicle['route_links']
        next_link_idx = vehicle['next_link_index'] + 1
        #(4) check if route is complete
        if next_link_idx >= len(route_links):
            updated_vehicles_df.at[idx, 'status'] = 'finished'
            continue
        #(5) get next link
        next_link = route_links[next_link_idx]
        #(6) check if next link can accept vehicle
        current_vehicles = len(vehicles_df[vehicles_df['current_link'] == next_link])
        max_vehicles = edges_df.loc[next_link, 'length_miles'] * edges_df.loc[next_link, 'lanes'] * edges_df.loc[next_link, 'jam_density_kj']
        if current_vehicles < max_vehicles:
            #(7) transition to next link
            updated_vehicles_df.at[idx, 'current_link'] = next_link
            updated_vehicles_df.at[idx, 'next_link_index'] = next_link_idx
            updated_vehicles_df.at[idx, 'position_on_link_meters'] = 0.0
        else:
            #(8) add to queue of current link
            updated_vehicles_df.at[idx, 'status'] = 'queued'
            updated_edges_df.at[current_link, 'queue_length'] += 1
    return updated_vehicles_df, updated_edges_df

# -----------------
# (5) Simulation control
# -----------------

def insert_new_vehicles(vehicles_df: pd.DataFrame, current_time_index: int) -> pd.DataFrame:
    #(1) make a copy to avoid modifying original
    updated_vehicles_df = vehicles_df.copy()
    #(2) find vehicles scheduled to depart at current time
    departing_mask = (vehicles_df['time_index'] == current_time_index) & (vehicles_df['status'] == 'waiting')
    #(3) activate these vehicles
    if departing_mask.any():
        updated_vehicles_df.loc[departing_mask, 'status'] = 'active'
        updated_vehicles_df.loc[departing_mask, 'position_on_link_meters'] = 0.0
        updated_vehicles_df.loc[departing_mask, 'accumulated_travel_time'] = 0.0
    return updated_vehicles_df

def run_simulation_timestep(vehicles_df: pd.DataFrame, edges_df: pd.DataFrame, current_time_index: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #(1) insert new vehicles for this timestep
    updated_vehicles_df = insert_new_vehicles(vehicles_df, current_time_index)
    #(2) calculate current network conditions
    updated_edges_df = calculate_link_conditions(edges_df, updated_vehicles_df)
    #(3) process queues and transitions
    updated_vehicles_df, updated_edges_df = process_link_transitions(updated_vehicles_df, updated_edges_df)
    updated_edges_df = manage_queues(updated_edges_df, updated_vehicles_df)
    #(4) update vehicle positions
    timestep = 5/60  # convert 5 minutes to hours
    updated_vehicles_df = update_vehicle_positions(updated_vehicles_df, updated_edges_df, timestep)
    #(5) print summary statistics
    active = (updated_vehicles_df['status'] == 'active').sum()
    queued = (updated_vehicles_df['status'] == 'queued').sum()
    finished = (updated_vehicles_df['status'] == 'finished').sum()
    waiting = (updated_vehicles_df['status'] == 'waiting').sum()
    print(f"time_index: {current_time_index}, active: {active}, queued: {queued}, finished: {finished}, waiting: {waiting}")
    return updated_vehicles_df, updated_edges_df

def run_simulation(vehicles_df: pd.DataFrame, edges_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    max_time_index = vehicles_df['time_index'].max() + 1
    #(1) initialize state
    current_vehicles_df = vehicles_df.copy()
    current_edges_df = edges_df.copy()
    #(2) run simulation for each timestep
    for time_index in range(max_time_index):
        current_vehicles_df, current_edges_df = run_simulation_timestep(
            current_vehicles_df, 
            current_edges_df,
            time_index
        )
        #(3) check if simulation is complete
        if (current_vehicles_df['status'] == 'waiting').sum() == 0 and \
           (current_vehicles_df['status'] == 'active').sum() == 0 and \
           (current_vehicles_df['status'] == 'queued').sum() == 0:
            print(f"simulation completed at time_index {time_index}")
            break
    return current_vehicles_df, current_edges_df