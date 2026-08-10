import os

from collections import defaultdict
from itertools import product

import networkx as nx
import pandas as pd

from .draw_graph import draw_graph
from .utils import save_graph_to_json


NOT_NODE_LABELS = (
    'WHNP', 'TO', 'WHADVP', 'CC', 'RP', 'PRT', 'IN', 'RB', 'MD', 'DT'
)

NODES_TO_EXCLUDE = {
    'The image', 'The message', 'This image', 
    "This", "The images", 'The scene', "the country",
    'There', "The narrative", "others", "something", '', 'part',
    'less', 'co', "A man", "Despite", 'she', 'her', 'they', 'them', 'he', 
    'him', 'his', 'its', 'it'
}


def create_graph(
        df: pd.DataFrame, 
        only_verbs_labels=True
    ) -> None:
    '''
    Creates a directed multigraph from parsed sentences in a DataFrame.

    Nodes represent words (excluding common stop words and pronouns),
    and edges represent actions or relations, primarily verbs.
    Each node and edge stores metadata: sentence IDs and media IDs.

    Args:
        df (pd.DataFrame): Must contain columns 'sentence_id', 'media_id',
                           'parsed_sentence', and 'parsed_labels'.
        only_verbs_labels (bool): If True, keeps only edges whose labels are verbs.

    Returns:
        nx.MultiDiGraph: Graph with nodes and edges annotated with metadata.
    '''

    G = nx.MultiDiGraph()
    node2metadata = defaultdict(lambda: {'sentence_ids': set(), 'media_ids': set()})

    def extract_edges(parsed_sentence, parsed_labels):
        end = len(parsed_labels) - 1  # last token is "."
        i = 0
        while i < end:
            j = i + 1
            edge_label = ''
            edge_pos = None

            while j < end:
                pos = parsed_labels[j]
                tok = parsed_sentence[j]

                if pos.startswith('V') or pos == 'MD':
                    edge_label = tok
                    edge_pos = pos
                elif pos.isalpha() and pos not in NOT_NODE_LABELS:
                    break
                else:
                    if not edge_label and pos != 'DT':
                        edge_label = tok
                    elif pos in ('RB', 'MD') or pos.startswith('V'):
                        edge_label += ' ' + tok
                j += 1

            if j < len(parsed_sentence):
                yield (
                    parsed_sentence[i],
                    edge_label,
                    parsed_sentence[j],
                    edge_pos or parsed_labels[j - 1]
                )

            i = j

    for row in df.itertuples(index=False):
        sentence_id = row.sentence_id
        media_id = row.media_id
        parsed_labels = row.parsed_labels
        parsed_sentence = row.parsed_sentence

        for node1, edge_label, node2, edge_pos in extract_edges(
            parsed_sentence, parsed_labels
        ):
            if {node1, node2} & NODES_TO_EXCLUDE:
                continue
            if any('their' in n.lower() for n in (node1, node2)):
                continue
            if edge_pos == 'ADVP':
                continue

            edge_found = False
            if G.has_edge(node1, node2):
                for _, data in G[node1][node2].items():
                    if data.get('label') == edge_label:
                        data['weight'] += 1
                        data['sentence_ids'].add(sentence_id)
                        data['media_ids'].add(media_id)
                        edge_found = True
                        break

            if not edge_found:
                G.add_edge(
                    node1,
                    node2,
                    label=edge_label,
                    weight=1,
                    sentence_ids={sentence_id},
                    media_ids={media_id},
                    pos=edge_pos
                )

            for node in (node1, node2):
                node2metadata[node]['sentence_ids'].add(sentence_id)
                node2metadata[node]['media_ids'].add(media_id)

    for node, meta in node2metadata.items():
        if node in G:
            G.nodes[node]['sentence_ids'] = list(meta['sentence_ids'])
            G.nodes[node]['media_ids'] = list(meta['media_ids'])

    for _, _, _, data in G.edges(keys=True, data=True):
        data['sentence_ids'] = list(data['sentence_ids'])
        data['media_ids'] = list(data['media_ids'])

    if only_verbs_labels:
        to_remove = [
            (u, v, k)
            for u, v, k, d in G.edges(keys=True, data=True)
            if not str(d.get('pos', '')).startswith('V')
        ]
        G.remove_edges_from(to_remove)
        G.remove_nodes_from(list(nx.isolates(G)))

    return G


def create_ego_graph(
        graph: nx.Graph, 
        center_node : str,
        egoless : bool = True
    ) -> nx.Graph:
    '''
    Create an ego graph centered on a specified node.
    Center node is excluded if egoless is True.
    
    Args:
        graph (nx.Graph): Original graph containing the center_node
        center_node (str): Node to center the ego graph around.
        egoless (bool): Bool whether or not to include the center node.

    Returns:
        nx.MultiDiGraph: Graph with nodes and edges annotated with metadata.
    '''
    ego_graph = nx.ego_graph(graph, center_node, radius=1)
    if egoless:
        ego_graph.remove_node(center_node)
    return ego_graph


def create_and_save_graph(
        df: pd.DataFrame, 
        output_dir:str, 
        event_type: str,
        user_type: str,
        only_verbs_labels:bool=True, 
        draw:bool=False
    ) -> nx.Graph:
    '''
    Create and save an graph in JSON format based on the Pandas DataFrame.
    Graphs are drawn and saved in HTML format if draw is True.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns 'sentence_id', 'media_id',
                           'parsed_sentence', and 'parsed_labels'.
        output_dir (str): Directory path where graph JSON and HTML files will be saved.
        event_type (str): List of events present in column 'event' to make graph representations.                               
        user_type (str): List of user types present in column 'event' to make graph representations.                               
        only_verbs_labels (bool): Flag passed to create_graph() to control labeling 
                                  behavior. Defaults to True
        draw (bool): Whether to draw and save the graphs as HTML files.
                            Defaults to False (can be super slow for big datasets).
    
    Returns:
        nx.MultiDiGraph: Created graph
    '''
    if df.empty:
        print("Warning: Empty dataframe, skipping...")
        return 
    
    graph = create_graph(df, only_verbs_labels=only_verbs_labels)
    print('Graph created with', len(graph.nodes), 'nodes and', len(graph.edges), 'edges.')
    output_path = os.path.join(output_dir, f'graph_{event_type}_{user_type}.json')
    save_graph_to_json(graph, path=output_path)
    print('Graph saved to:', output_path)
    
    if draw:
        output_filename = os.path.join(output_dir, f'graph_{event_type}_{user_type}.html')
        draw_graph(
            graph,
            output_filename=output_filename
        )
        print('Graph saved to:', output_filename)
    return graph


def create_and_save_multiple_graphs(
        df: pd.DataFrame, 
        output_dir:str, 
        event_types: list[str],
        user_types: list[str],
        only_verbs_labels:bool=True, 
        draw:bool=False
    ) -> dict:
    '''
    Creates and saves graph representations in JSON format for all combinations of event en user types.
    Graphs are drawn and saved in HTML format if draw_graphs is true.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns 'sentence_id', 'media_id',
                           'parsed_sentence', and 'parsed_labels'.
        output_dir (str): Directory path where graph JSON and HTML files will be saved.
        event_type (list[str]): List of events present in column 'event' to make graph representations.                             
        user_type (list[str]): List of user types present in column 'event' to make graph representations.                               
        only_verbs_labels (bool): Flag passed to create_graph() to control labeling 
                                  behavior. Defaults to True
        draw (bool): Whether to draw and save the graphs as HTML files.
                            Defaults to False (can be super slow for big datasets).
    
    Returns:
        nx.MultiDiGraph: Created graphs
    '''
    name2graph = {}
    combinations = list(product(event_types, user_types))
    for i, (event_type,user_type) in enumerate(combinations):
        subset = df.copy()
        if event_type is not None:
            subset = subset[subset['event'] == event_type]
        if user_type is not None:
            subset = subset[subset['usr_type'] == user_type]

        print(f"Creating graph {i+1}/{len(combinations)}...")
        print(f"Event: {event_type}, User type: {user_type}, Number of sentences: {len(subset)}")

        if subset.empty:
            print("Warning: Empty dataframe found, skipping...")
            continue

        graph = create_and_save_graph(subset,output_dir,event_type,user_type,only_verbs_labels,draw)
        name2graph[f'{event_type}_{user_type}'] = graph
    
    return name2graph


def create_and_save_ego_graph(
        graph: nx.Graph,
        center_node: str,
        output_dir: str,
        event_type: str,
        user_type: str,
        egoless : bool = True,
        draw : bool = False
    ) -> nx.Graph:  
    """
    Create and save an ego graphs in JSON format based on graph.
    Center node is not included if egoless is True.
    Graphs are drawn and saved in HTML format if draw_graphs is true.
    
    Args:
        graph (nx.Graph): The source graph object to extract the ego network from.
        center_node (str): The central node for the ego network extraction.
        output_dir (str): Directory path where graph JSON and HTML files will be saved.
        event_types (str): List of events present in column 'event' to make graph representations.                               
        user_types (str): List of user types present in column 'event' to make graph representations.    
        egoless (bool) : Bool whether or not to include the center node.                           
        draw (bool): Whether to draw and save the graphs as HTML files.
                    Defaults to False (can be super slow for big datasets).
    
    Returns:
        nx.MultiDiGraph : The generated ego graph (without center node).
    """
    if graph.number_of_nodes() == 0:
        print("Warning: Empty graph, skipping...")
        return 
    if center_node not in graph:
        raise ValueError(f"Center node '{center_node}' not found in graph. "
                        f"Example available nodes: {list(graph.nodes())[:10]}...")
    
    graph = create_ego_graph(graph, center_node, egoless)
    print('Ego graph created with', len(graph.nodes), 'nodes and', len(graph.edges), 'edges.')
    output_path = os.path.join(output_dir, f'ego_graph_{center_node}_{event_type}_{user_type}.json')
    save_graph_to_json(graph, path=output_path)
    print('Graph saved to:', output_path)
    
    if draw:
        output_path_visual = os.path.join(output_dir, f'ego_graph_{center_node}_{event_type}_{user_type}.html')
        draw_graph(
            graph,
            output_filename=output_path_visual
        )
        print('Graph saved to:', output_path_visual)
    return graph


def create_and_save_multiple_ego_graphs(
        name2graph: dict[str, nx.Graph], 
        center_node,
        egoless: str, 
        output_dir: str,
        draw : bool = False
    ) -> dict[str, nx.Graph]:
    """
    Create and save multiple ego graphs in JSON format based on the graphs in name2graph.
    Center node is not included if egoless is True.
    Graphs are drawn and saved in HTML format if draw_graphs is true.
    
    Args:
        graph (nx.Graph): The source graph object to extract the ego network from.
        center_node (str): The central node for the ego network extraction.
        output_dir (str): Directory path where graph JSON and HTML files will be saved.
        egoless (bool) : Bool whether or not to include the center node.
        event_types (str): List of events present in column 'event' to make graph representations.                               
        user_types (str): List of user types present in column 'event' to make graph representations.                               
        draw (bool): Whether to draw and save the graphs as HTML files.
                    Defaults to False (can be super slow for big datasets).
    
    Returns:
        nx.MultiDiGraph : The generated ego graph (without center node).
    """
    name2egograph = {}
    for i, ((event_type, user_type), graph) in enumerate(name2graph.items()):
        print(f"Creating graph {i+1}/{len(name2graph)}...")
        if graph.number_of_nodes() == 0:
            print("Warning: Empty dataframe found, skipping...")
            continue
        print(f"Event: {event_type}, User type: {user_type}")
        graph = create_and_save_ego_graph(graph,center_node,output_dir,egoless,event_type,user_type,draw)
        name2egograph[f'{event_type}_{user_type}'] = graph
    
    return name2egograph
