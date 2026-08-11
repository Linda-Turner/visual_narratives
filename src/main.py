import ast
import os

import pandas as pd

from preprocessing.orchestrator import preprocess_sentences
from clustering.orchestrator import (
    cluster_verb_and_noun_phrases,
    update_sentences_with_clusterized)
from make_graphs.build_graph import create_and_save_multiple_graphs
from make_graphs.draw_graph import draw_graph

def build_visual_narratives(input_file: str, output_dir: str):
    '''
    The complete visual narrative generation pipeline from step 2 descriptions, 
    including preprocessing, syntactic parsing, phrase clustering, and graph construction.

    Args:
        Input_file (str): .tsv file with columns: 'Dir', 'ImageID', 'Labels'. 
            'Labels' contain step 2 descriptions generated in earlier stages in the pipeline.
        Output_dir (str): directory to save the narratives
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    

    # Preprocess
    data_output_dir = os.path.join(output_dir, "data")
    if not os.path.exists(data_output_dir):
        os.makedirs(data_output_dir)
    preprocessed_sentences_path = preprocess_sentences(input_file, data_output_dir)

    preprocessed_df = pd.read_csv(
        preprocessed_sentences_path,
        converters={
            'parsed_labels': ast.literal_eval,
            'parsed_sentence': ast.literal_eval
        }
    )

    # Clustering
    clustering_output_dir = os.path.join(output_dir, "clustering")
    if not os.path.exists(clustering_output_dir):
        os.makedirs(clustering_output_dir)
    cluster_verb_and_noun_phrases(preprocessed_df, clustering_output_dir)

    # Update sentences and merg with data 
    updated_roles_df = update_sentences_with_clusterized(preprocessed_df, clustering_output_dir)
    updated_roles_df['path'] = updated_roles_df['Dir'].astype(str) + "/" + updated_roles_df['ImageID'].astype(str)
    original_data = pd.read_csv(f'data/data.tsv', sep='\t')
    merged_df = updated_roles_df.merge(original_data, on="path", how="inner")
    merged_df.to_csv(os.path.join(data_output_dir, 'updated_data.csv'), index=False)

    # Create grapgh
    graphs_output_dir = os.path.join(output_dir, "graphs")
    if not os.path.exists(graphs_output_dir):
        os.makedirs(graphs_output_dir)
    name2graph = create_and_save_multiple_graphs(merged_df, graphs_output_dir,['cop', 'strike'],['m', 'c'])

    # Draw graph
    for name, graph in name2graph.items():
        if name.endswith('_c'):
            draw_graph(
                graph,
                output_filename=os.path.join(graphs_output_dir, f'{name}_graph.html')
            )

if __name__ == "__main__":
    INPUT_FILE = 'data/data_test.tsv'  # input .tsv file with columns: 'Dir', 'ImageID', 'Labels'
    OUTPUT_DIR = 'output_narratives'  # directory to save the narratives

    build_visual_narratives(INPUT_FILE, OUTPUT_DIR)

# def parse_args():
#     """Parse command-line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Generate visual narratives from an input TSV file."
#     )

#     parser.add_argument(
#         "input_file",
#         help="Path to the input TSV file containing the columns 'Dir', 'ImageID', and 'Labels'."
#     )

#     parser.add_argument(
#         "-o", "--output-dir",
#         default="output_narratives",
#         help="Directory where generated files will be saved."
#     )

#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     build_narratives(args.input_file, args.output_dir)