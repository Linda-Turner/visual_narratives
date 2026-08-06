import ast
import os

import pandas as pd

from preprocessing.orchestrator import preprocess_sentences
from clustering.orchestrator import (
    cluster_verb_noun_phrases,
    update_sentences_with_clusterized)
from make_graphs.utils import split_df_create_graphs, draw_graph

PCA_ARGS = {'n_components': 50, 'svd_solver': 'full'}

def build_visual_narratives(input_file: str, output_dir: str, pca_args: dict=PCA_ARGS):
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
    cluster_verb_noun_phrases(preprocessed_df, clustering_output_dir, pca_args, 50)

    # Update sentences and merg with data 
    preprocessed_df['path'] = preprocessed_df['Dir'].astype(str) + "/" + preprocessed_df['ImageID'].astype(str)
    updated_roles_df = update_sentences_with_clusterized(preprocessed_df, clustering_output_dir)
    data = pd.read_csv(f'data/data.tsv', sep='\t')
    merged_df = updated_roles_df.merge(data, on="path", how="inner")
    merged_df.to_csv(os.path.join(data_output_dir, 'updated_data.csv'), index=False)

    # Draw grapgh
    name2graph = split_df_create_graphs(merged_df, output_dir)
    for name, graph in name2graph.items():
        if name.endswith('_c'):
            draw_graph(
                graph,
                output_filename=os.path.join(output_dir, f'{name}_graph.html')
            )
            print('Graph is drawn and saved to:', os.path.join(output_dir, f'{name}_graph.html'))

        

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