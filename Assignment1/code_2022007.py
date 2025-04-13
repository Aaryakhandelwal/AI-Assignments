import numpy as np
import pickle
from memory_profiler import memory_usage
import heapq
import math
import gc

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.

#intializing global variables
global path_exists
path_exists = None

class Node:
    def __init__(self, state, h_score=0, parent=None, depth=0, path_cost=0, x=0, y=0, f_score=0):
        self.state = state
        self.parent = parent
        self.depth = depth
        self.path_cost = path_cost

        # Used only in case of A* algorithms
        self.x = x
        self.y = y
        self.h_score = h_score
        self.f_score = f_score

    def expand(self, adj_matrix, rev=False):
        # Expands a node and allocates memory for each child node
        children = []
        node = self.state
        for child, cost in enumerate(adj_matrix[node]):
            if cost != 0:
                children.append(Node(child, parent=self, depth=self.depth + 1, path_cost=self.path_cost + cost))
        if rev:
            children.reverse()
        return children

    # Used for A* algorithms
    def expand_heuristic(self, adj_matrix, node_attributes, start_node, goal_node):
        children = []
        node = self.state
        for child_state, cost in enumerate(adj_matrix[node]):
            if cost != 0:
                h_score = calculate_heuristic(node_attributes, start_node, goal_node, child_state)
                child = Node(
                    state=child_state,
                    h_score=h_score,
                    parent=self,
                    depth=self.depth + 1,
                    path_cost=self.path_cost + cost,
                    x=node_attributes[child_state]['x'],
                    y=node_attributes[child_state]['y'],
                    f_score=self.path_cost + h_score
                )
                children.append(child)

        return children

    def __lt__(self, other):
        # Used in the A* algorithm, for inserting the nodes in a priority queue based on heuristic
        return self.f_score < other.f_score


def floyd_warshall(adj_matrix):
    n = len(adj_matrix)

    # Initialize a matrix for path existence (True if path exists, False otherwise)
    path_matrix = np.array(adj_matrix, dtype=bool)

    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                path_matrix[i][j] = path_matrix[i][j] or (path_matrix[i][k] and path_matrix[k][j])

    return path_matrix

# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]




def in_cycle(node):
    # Checks if the node exists in a cycle
    ancestor_node = node.parent
    while ancestor_node is not None:
        if node.state == ancestor_node.state:
            return True
        ancestor_node = ancestor_node.parent
    return False


def find_path(node):
    # Given a node, backtracks the path using parent of each node
    path = []
    while node is not None:
        path.append(node.state)
        node = node.parent
    path.reverse()
    return path


def dfs(adj_matrix, start_node, goal_node, depth):
    # Applies the Depth First Search algorithm for a given depth.
    frontier = [Node(start_node)]
    cutoff_occurred = False

    while frontier:
        node = frontier.pop()

        if node.state == goal_node:  # Returns if the goal state is found
            return node

        if node.depth > depth:  # Checks for depth/threshold
            cutoff_occurred = True
        else:
            # if node.state not in visited:
                # visited.add(node.state)  # Mark node as visited at this depth level
            for child in node.expand(adj_matrix, rev=True):
                if not in_cycle(child):
                    frontier.append(child)

    return 'cutoff' if cutoff_occurred else None


def get_ids_path(adj_matrix, start_node, goal_node):

    global path_exists

    if path_exists is None:
        path_exists = floyd_warshall(adj_matrix)

    if not path_exists[start_node][goal_node]:
        return None

    depth = 0
    max_depth = len(adj_matrix)

    while depth < max_depth:
        # visited = set()  # Reset visited set at each depth level
        result = dfs(adj_matrix, start_node, goal_node, depth)

        if result == 'cutoff':
            depth += 1
        elif result is None:
            return None
        else:
            return find_path(result)

    return None  # If no solution is found after max depth


# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def join_nodes(node_f, node_b):
    # Finds the path given the forward frontier and backward frontier
    path = []
    while node_f:
        path.append(node_f.state)
        node_f = node_f.parent
    path.reverse()

    # Collect the backward path
    node_b_path = []
    node_b = node_b.parent
    while node_b:
        node_b_path.append(node_b.state)
        node_b = node_b.parent

    # Combine paths
    path.extend(node_b_path)
    return path

def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    # initialization
    node_f = Node(start_node)
    node_b = Node(goal_node)
    frontier_f = [node_f]
    frontier_b = [node_b]
    visited_f = {node_f.state : node_f}
    visited_b = {node_b.state : node_b}

    while frontier_f and frontier_b:
        # forward search
        nf = frontier_f[0]
        frontier_f = frontier_f[1:]
        if (nf.state in visited_b):
            return join_nodes(nf, visited_b[nf.state])
        for child in nf.expand(adj_matrix):
            # if child.state not in visited_f or child.path_cost < visited_f[child.state].path_cost:
            if child.state not in visited_f:
                visited_f[child.state] = child
                frontier_f.append(child)

        # backward search

        nb = frontier_b[0]
        frontier_b = frontier_b[1:]
        if (nb.state in visited_f):
            return join_nodes(visited_f[nb.state], nb)
        for child in nb.expand(adj_matrix):
            # if child.state not in visited_b or child.path_cost < visited_b[child.state].path_cost:
            if child.state not in visited_b:
                visited_b[child.state] = child
                frontier_b.append(child)
    gc.collect()
    return None




# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def dist(node_attributes, node1, node2):
    node1_x = node_attributes[node1]['x']
    node1_y = node_attributes[node1]['y']
    node2_x = node_attributes[node2]['x']
    node2_y = node_attributes[node2]['y']
    return math.sqrt((node1_x - node2_x) ** 2 + (node1_y - node2_y) ** 2)

def calculate_heuristic(node_attributes, start_node, goal_node, node):
    return dist(node_attributes, start_node, node) + dist(node_attributes, node, goal_node)

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    start = Node(
        start_node,
        x=node_attributes[start_node]['x'],
        y=node_attributes[start_node]['y'],
        h_score=calculate_heuristic(node_attributes, start_node, goal_node, start_node)
    )
    goal = Node(
        goal_node,
        x=node_attributes[goal_node]['x'],
        y=node_attributes[goal_node]['y'],
        h_score=calculate_heuristic(node_attributes, start_node, goal_node, goal_node)
    )

    nodes_to_be_evaluated = []
    heapq.heappush(nodes_to_be_evaluated, start)

    nodes_expanded = set()

    while nodes_to_be_evaluated:
        node = heapq.heappop(nodes_to_be_evaluated)
        if node.state == goal.state:
            return find_path(node)
        nodes_expanded.add(node.state)

        for child in node.expand_heuristic(adj_matrix, node_attributes, start_node, goal_node):
            tentative_path_cost = node.path_cost + adj_matrix[node.state][child.state]

            # If child not yet expanded or a shorter path to child is found
            if child.state not in nodes_expanded or tentative_path_cost < child.path_cost:
                child.path_cost = tentative_path_cost
                child.f_score = child.path_cost + child.h_score
                heapq.heappush(nodes_to_be_evaluated, child)


    return None


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):

    start_f = Node(
        state=start_node,
        x=node_attributes[start_node]['x'],
        y=node_attributes[start_node]['y'],
        h_score=calculate_heuristic(node_attributes, start_node, goal_node, start_node)
    )
    goal_b = Node(
        state=goal_node,
        x=node_attributes[goal_node]['x'],
        y=node_attributes[goal_node]['y'],
        h_score=calculate_heuristic(node_attributes, start_node, goal_node, goal_node)
    )

    frontier_f = []
    frontier_b = []
    heapq.heappush(frontier_f, (start_f.f_score, start_f))
    heapq.heappush(frontier_b, (goal_b.f_score, goal_b))

    visited_f = {start_f.state: start_f}
    visited_b = {goal_b.state: goal_b}

    while frontier_f and frontier_b:
        # Forward search
        f_score, nf = heapq.heappop(frontier_f)
        if nf.state in visited_b:
            return join_nodes(nf, visited_b[nf.state])

        for child in nf.expand_heuristic(adj_matrix, node_attributes, start_node, goal_node):
            tentative_path_cost = nf.path_cost + adj_matrix[nf.state][child.state]
            if child.state not in visited_f or tentative_path_cost < child.path_cost:
                child.path_cost = tentative_path_cost
                child.f_score = child.path_cost + child.h_score
                visited_f[child.state] = child
                heapq.heappush(frontier_f, (child.f_score, child))

        # Backward search
        b_score, nb = heapq.heappop(frontier_b)
        if nb.state in visited_f:
            return join_nodes(visited_f[nb.state], nb)

        for child in nb.expand_heuristic(adj_matrix, node_attributes, start_node, goal_node):
            tentative_path_cost = nb.path_cost + adj_matrix[nb.state][child.state]
            if child.state not in visited_b or tentative_path_cost < child.path_cost:
                child.path_cost = tentative_path_cost
                child.f_score = child.path_cost + child.h_score
                visited_b[child.state] = child
                heapq.heappush(frontier_b, (child.f_score, child))

    return None




# Make sure to define or import your `get_astar_search_path` and `get_bidirectional_astar_search_path` functions


# Bonus Problem

# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].


def tarjans_algorithm(undirected_adj_matrix):

    num_nodes = len(undirected_adj_matrix)

    adjacency_list = [[] for _ in range(num_nodes)]

    # Convert adjacency matrix to adjacency list
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] != 0:
                adjacency_list[i].append(j)


    discovery = [num_nodes] * num_nodes
    low = [num_nodes] * num_nodes

    bridges = []

    def dfs(current_node, discovery_time, parent_node):
        # If this node has not been visited
        if discovery[current_node] == num_nodes:

            discovery[current_node] = discovery_time
            low[current_node] = discovery_time
            
            for neighbor in adjacency_list[current_node]:

                if neighbor == parent_node:
                    continue

                neighbor_discovery_time = discovery_time + 1
                neighbor_low_value = dfs(neighbor, neighbor_discovery_time, current_node)

                # Check if the edge is a bridge
                if neighbor_low_value > discovery[current_node]:
                    bridges.append([current_node, neighbor])

                # Update the low value of the current node
                low[current_node] = min(low[current_node], neighbor_low_value)

        return low[current_node]

    # Start DFS from node 0 (or any node if the graph is connected)
    for node in range(num_nodes):
        if discovery[node] == num_nodes:
            dfs(node, 0, -1)

    return bridges

def convert_to_undirected_graph(adj_matrix):
    num_nodes = len(adj_matrix)
    undirected_adj_matrix = [row[:] for row in adj_matrix]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] != 0 and i != j:
                undirected_adj_matrix[j][i] = 1
                undirected_adj_matrix[i][j] = 1

    return undirected_adj_matrix

def bonus_problem(adj_matrix):
    undirected_adj_matrix = convert_to_undirected_graph(adj_matrix)
    edges = tarjans_algorithm(undirected_adj_matrix)
    return edges


if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)


  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')

