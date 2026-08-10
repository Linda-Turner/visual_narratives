from copy import deepcopy

import networkx as nx
from networkx.algorithms.community import louvain_communities


def get_strength(
        graph : nx.Graph, 
        topn : int = 20
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    '''
        Returns the top n nodes ranked by weighted degree (strength),
        weighted in-degree (in-strength), and weighted out-degree
        (out-strength).

        Args:
            graph (nx.Graph): NetworkX graph for which node strengths
                should be calculated. 
            topn (int): Number of top nodes to return for each strength
                measure. Defaults to 20.

        Returns:
            tuple[dict[str, int], dict[str, int], dict[str, int]]:
                A tuple containing three dictionaries:
                - strength: Top n nodes ranked by weighted degree.
                - in_strength: Top n nodes ranked by weighted in-degree.
                - out_strength: Top n nodes ranked by weighted out-degree.
            Strength values are non-normalized weighted degrees.
    '''
    def _sort_di_view(item):
        return dict(sorted(item, key=lambda x: x[1], reverse=True)[:topn])
    strength = _sort_di_view(graph.degree(weight='weight'))
    in_strength = _sort_di_view(graph.in_degree(weight='weight'))
    out_strength = _sort_di_view(graph.out_degree(weight='weight'))
    return strength, in_strength, out_strength


def get_betweenness(
        graph : nx.Graph, 
        topn : int = 20
    ) -> dict[str, float]:
    '''
    Returns the top n nodes ranked by betweenness centrality.

    The returned values are normalized and sorted in descending
    order of betweenness centrality.

    Note:
        Calculating betweenness centrality can be computationally
        expensive for large graphs.

    Args:
        graph (nx.Graph): NetworkX graph for which betweenness
            centrality should be calculated. 
        topn (int): Number of top nodes to return. Defaults to 20.

    Returns:
        dict[str, float]: Dictionary containing the top n nodes and
            their normalized betweenness centrality values, sorted in
            descending order.
    '''
    betweenness = nx.betweenness_centrality(graph, weight='weight')
    return dict(sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:topn])


def get_strength_and_betweenness(
        graph: nx.Graph, 
        topn : int = 20
    ) -> tuple[dict, dict, dict, dict]:
    '''
    Returns strength, in_strength, out_strength, betweenness.

    Args:
        graph (nx.Graph): NetworkX graph for which the measures
            should be calculated. 
        topn (int): Number of top nodes to return. Defaults to 20.

    Returns:
        tuple[dict, dict, dict, dict]:
            A tuple containing four dictionaries:
                - strength, in_strength, out_strength, betweenness

            Strength values are non-normalized weighted degrees.
            Betweenness centrality values are normalized.
    '''
    strength, in_strength, out_strength = get_strength(graph, topn=topn)
    betweenness = get_betweenness(graph, topn=topn)
    return strength, in_strength, out_strength, betweenness


def get_strength_betweenness_multiple(
        name2graph : dict[str, nx.Graph], 
        topn : int = 20
    ) -> dict:
    '''
    Strength and betweenness for multiple graphs.

    Args:
        graph (nx.Graph): NetworkX graph for which the measures
            should be calculated. 
        topn (int): Number of top nodes to return. Defaults to 20.

    Returns:
        dict: Dictionary mapping each graph name to a dictionary containing:
            strength, in_strength, out_strength, betweenness

            Strength values are non-normalized weighted degrees.
            Betweenness centrality values are normalized.
    '''
    name2metrics_dict = {}
    for name, graph in name2graph.items():
        metrics = get_strength_and_betweenness(graph, topn=topn)
        name2metrics_dict[name] = {
            'strength': metrics[0],
            'in_strength': metrics[1],
            'out_strength': metrics[2],
            'betweenness': metrics[3]
        }
    return name2metrics_dict


def print_metrics_table_for_graph(
        metrics: tuple[dict], 
        with_numbers: bool = False
    ) -> None:
    '''
    Print a metrics table for a single graph.

    Args:
        metrics: Tuple of four dicts (strength, in_strength, out_strength, betweenness).
        with_numbers: If True, include metric values; otherwise print ranked nodes only.
    '''
    metrics = {
        'strength': metrics[0],
        'in_strength': metrics[1],
        'out_strength': metrics[2],
        'betweenness': metrics[3]
    }
    columns = list(metrics.keys())
    header = " ".join(f"{col:<30}" for col in columns)
    print(header)
    print("-" * 120)

    all_keys = [list(metrics[col].keys()) for col in columns]
    max_len = max(len(keys) for keys in all_keys)

    for i in range(max_len):
        row_items = []
        for col_idx, col in enumerate(columns):
            keys = all_keys[col_idx]
            if i < len(keys):
                key = keys[i]
                if with_numbers:
                    item = f"{key}: {metrics[col][key]}"
                    row_items.append(f"{item:<30}")
                else:
                    row_items.append(f"{key:<30}")
            else:
                row_items.append(f"{'':<30}")
        print(" ".join(row_items))


def extract_communities(
        graph : nx.Graph, 
        seed : int = 42
    ) -> list[set]:
    '''
    Detect communities in a graph using the Louvain method.

    Args:
        graph (nx.Graph): NetworkX graph on which to perform community
            detection. 
        seed (int): Random seed used by the Louvain algorithm to ensure
            reproducible community assignments. Defaults to 42.

    Returns:
        list[set]: a list of sets, where each set contains the nodes belonging to a community.

    '''
    return louvain_communities(graph, weight='weight', seed=seed)


def make_subgraph_from_community(
        graph : nx.Graph, 
        community : set
    ) -> nx.Graph:
    """
    Create a subgraph containing only nodes from the specified community.

    Args:
        graph (nx.Graph): NetworkX graph from which the subgraph is extracted.
        community (set): Set of node identifiers defining the community.

    Returns:
        nx.Graph: A subgraph containing only nodes and the edges between them included in the community.
    """
    community_graph = graph.subgraph(community).copy()
    return community_graph


def k_core_weighted_multigraph(
        graph: nx.Graph, 
        k : int = 2 
    ) -> nx.Graph:
    '''
    Extract the k-core of a weighted multigraph by iteratively removing nodes
    with weighted degree less than k.

    Args:
        graph (nx.Graph): NetworkX graph from which the weighted k-core is
            extracted. 
        k (int): Minimum weighted degree threshold for nodes to remain in the
            graph. Nodes with weighted degree below this value are removed.
            Defaults to 2.

    Returns:
        nx.Graph: A subgraph containing only nodes that satisfy
            the weighted degree threshold after iterative pruning.
    '''
    new_graph = deepcopy(graph)
    changed = True
    while changed:
        changed = False
        # "weight" is supposed to sum the weights of parallel edges
        to_remove = [n for n in new_graph if new_graph.degree(n, weight='weight') < k]
        if to_remove:
            new_graph.remove_nodes_from(to_remove)
            changed = True
    return new_graph