import json

import networkx as nx

def save_graph_to_json(graph: nx.Graph, path: str):
    """
    Save a NetworkX graph to a JSON file.

    Args:
        graph (nx.Graph): NetworkX graph to save.
        path (str): Path to the output JSON file.
    """
    data = nx.node_link_data(graph, edges="edges")
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def read_graph_from_json(path: str) -> nx.Graph:
    """
    Load a NetworkX graph from a JSON file.

    Args:
        path (str): Path to the JSON file containing the graph.

    Returns:
        nx.Graph: NetworkX graph reconstructed from the JSON data.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")