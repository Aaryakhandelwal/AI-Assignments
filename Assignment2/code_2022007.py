# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """

    # converting data types as required 
    df_fare_attributes["fare_id"] = df_fare_attributes["fare_id"].astype(str)
    df_fare_attributes["currency_type"] = df_fare_attributes["currency_type"].astype(str)
    df_fare_attributes["agency_id"] = df_fare_attributes["agency_id"].astype(str)
    df_fare_attributes["old_fare_id"] = df_fare_attributes["old_fare_id"].astype(str)
    df_fare_rules["fare_id"] = df_fare_rules["fare_id"].astype(str)
    df_routes["agency_id"] = df_routes["agency_id"].astype(str)
    df_routes["route_long_name"] = df_routes["route_long_name"].astype(str)
    df_stop_times["trip_id"] = df_stop_times["trip_id"].astype(str)
    df_stop_times["arrival_time"] = df_stop_times["arrival_time"].astype('timedelta64[s]')
    df_stop_times["departure_time"] = df_stop_times["departure_time"].astype('timedelta64[s]')
    df_stops["stop_code"] = df_stops["stop_code"].astype(str)
    
    # checking if changed data types are seen
    # print(df_fare_attributes.dtypes)
    # print(df_fare_rules.dtypes)
    # print(df_routes.dtypes)
    # print(df_trips.dtypes)
    # print(df_stop_times.dtypes)
    # print(df_stops.dtypes)

    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mappings

    trip_to_route = df_trips.set_index("trip_id")["route_id"].to_dict()
    # Map route_id to a list of stops in order of their sequence
          
    merge_stop_times_and_trips = df_stop_times.merge(df_trips, on="trip_id")
    merge_stop_times_and_trips = merge_stop_times_and_trips.sort_values(by=["route_id", "trip_id", "stop_sequence"])
        
    # Ensure each route only has unique stops

    route_to_stops = (
        merge_stop_times_and_trips.groupby("route_id")["stop_id"]
        .apply(lambda stops: list(stops.drop_duplicates()))
        .to_dict()
    )


    # Count trips per stop
    stop_trip_count = df_stop_times["stop_id"].value_counts().to_dict()

    # Create fare rules for routes
    fare_rules = defaultdict(list)
    fare_rules = (
        df_fare_rules.groupby("route_id")
        .apply(lambda x: x[["fare_id", "origin_id", "destination_id"]].to_dict(orient="records"))
        .to_dict()
    )


    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = df_fare_rules.merge(df_fare_attributes, on = "fare_id")
    
# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    route_trip_counts = defaultdict(int)
    for trip_id, route_id in trip_to_route.items():
        route_trip_counts[route_id] += 1

    busiest_routes = sorted(route_trip_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(busiest_routes)
    return busiest_routes  # Implementation here

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    frequent_stops = sorted(stop_trip_count.items(), key = lambda x: x[1], reverse =  True)[:5]
    print(frequent_stops)
    return frequent_stops

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    stop_route_counts = defaultdict(int)

    for route_id, stops in route_to_stops.items():
        for stop_id in set(stops):  
            stop_route_counts[stop_id] += 1

    busiest_stops = sorted(stop_route_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(busiest_stops)
    
    return busiest_stops    # Implementation here

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    stop_pair_route_count = defaultdict(set) 

    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            stopA, stopB = stops[i], stops[i + 1]
            stop_pair = tuple((stopA, stopB))
            stop_pair_route_count[stop_pair].add(route_id)
    

    single_route_pairs = []
    for stop_pair, routes in stop_pair_route_count.items():
        if (len(routes) == 1):
            combined_frequency = stop_trip_count[stop_pair[0]] + stop_trip_count[stop_pair[1]]
            single_route_pairs.append((stop_pair, next(iter(routes)), combined_frequency))

    top_5_stop_pairs = sorted(single_route_pairs, key = lambda x: x[2], reverse = True)[:5]
    top_5_stop_pairs = [(i[0], i[1]) for i in top_5_stop_pairs]

    print(top_5_stop_pairs)
    return top_5_stop_pairs


# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    graph = nx.DiGraph()
    for stops in route_to_stops.values():
        graph.add_edges_from((stops[i], stops[i + 1]) for i in range(len(stops) - 1))

    layout_3d = nx.spring_layout(graph, dim=3)

    x_edges, y_edges, z_edges = [], [], []
    for src, dst in graph.edges():
        x_edges += [layout_3d[src][0], layout_3d[dst][0], None]
        y_edges += [layout_3d[src][1], layout_3d[dst][1], None]
        z_edges += [layout_3d[src][2], layout_3d[dst][2], None]

    edge_trace = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(width=0.75, color='blue'),
        hoverinfo='none'
    )

    x_nodes, y_nodes, z_nodes, labels = [], [], [], []
    for node in graph.nodes():
        x, y, z = layout_3d[node]
        x_nodes.append(x)
        y_nodes.append(y)
        z_nodes.append(z)
        labels.append(f'Stop ID: {node}')

    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        marker=dict(size=7, color='blue'),
        text=labels,
        hoverinfo='text'
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='3D Route to Stops Network Graph',
            showlegend=False,
            margin=dict(b=0, l=0, r=0, t=40),
            scene=dict(
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            )
        )
    )

    fig.show()

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    route_ids_list = []
    for route_id, stops in route_to_stops.items():
        if (start_stop in stops and end_stop in stops) :
            route_ids_list.append(route_id)


    return route_ids_list  # Implementation here

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('PDDL, BoardRoute, TransferRoute, RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: PDDL, BoardRoute, TransferRoute, RouteHasStop, DirectRoute, OptimalRoute")  # Confirmation print

    # Define Datalog predicates

    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog
    
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops)):
            + RouteHasStop(route_id, stops[i])

    # for direct routes part b)
    DirectRoute(X, Y, R) <= (
        RouteHasStop(R, X) &
        RouteHasStop(R, Y)
    )

    # for forward and backward chaining c)
    OptimalRoute(X, Y, Z, R1, R2) <= (
        DirectRoute(X, Z, R1) &
        DirectRoute(Z, Y, R2) &
        (R1 != R2)  
    )

    # for pddl : bonus part
    BoardRoute(X, R1) <= (
        RouteHasStop(R1, X)
    )

    TransferRoute(X, R1, R2) <= (
        RouteHasStop(R1, X) & 
        RouteHasStop(R2, X) & 
        (R1 != R2)
    )

    PDDL(X, Y, Z, R1, R2) <= (
        BoardRoute(X, R1) & 
        TransferRoute(Y, R1, R2) & 
        BoardRoute(Z, R2)
    )

        
# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (str): The ID of the starting stop.
        end (str): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """

    query = DirectRoute(start, end, R)
    return sorted([route_id[0] for route_id in query])


# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    

    query = OptimalRoute(start_stop_id, end_stop_id, stop_id_to_include, R1, R2)
    return [(route1, stop_id_to_include, route2) for route1, route2 in query]

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """

    query = OptimalRoute(end_stop_id, start_stop_id, stop_id_to_include, R1, R2)
    return [(route1, stop_id_to_include, route2) for route1, route2 in query]



# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.

        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """

    query = PDDL(start_stop_id, stop_id_to_include, end_stop_id, R1, R2)
    return [(route1, stop_id_to_include, route2) for route1, route2 in query]


# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pass  # Implementation here

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    pass  # Implementation here

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    pass  # Implementation here
