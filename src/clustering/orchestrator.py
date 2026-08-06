import os
import ast

import pandas as pd

from .clusterize import (
    prepare_df_to_clustering,
    clusters_and_write
)

def cluster_verb_noun_phrases(
    df: pd.DataFrame, 
    output_dir: str, 
    pca_args: dict = {'n_components': 50, 'svd_solver': 'full'},
    batch_size=15000
    ) -> None:
    '''
    A wrapper function to clusterize verbs and noun phrases from the Pandas DataFrame.
    Clustering is performed separately for verbs and noun phrases.
    The function saves the clustering results in the specified output directory.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing syntaxically parsed sentences.
        output_dir (str): Path to the folder containing the clustering results.
        pca_args (dict): Dictionary of PCA parameters. Defaults to 50 components with 'full' SVD solver.
        batch_size (int): Maximum number of phrases per batch for clustering. Defaults to 15000.
    '''
    cluster_dirs = {
        "verbs": os.path.join(output_dir, "verbs"),
        "noun_phrases": os.path.join(output_dir, "noun_phrases"),
    }
    for directory in cluster_dirs.values():
        os.makedirs(directory, exist_ok=True)

    # Prepare documents for clustering for verbs and nou phrases separately from the DataFrame
    verbs_path, noun_phrases_path = prepare_df_to_clustering(
        df,
        verb_dir=cluster_dirs["verbs"],
        noun_phrases_dir=cluster_dirs["noun_phrases"],
    )

    # Cluster verbs and noun phrases seperately
    cluster_paths = {
        "Verbs": verbs_path,
        "Noun_phrases": noun_phrases_path,
    }
    for name, path in cluster_paths.items():
        print(f"\n{'='*60}")
        print(f"Clustering {name.lower()}...")

        clusters_and_write(
            path,
            cluster_dirs[name.lower()],
            pca_args=pca_args,
            batch_size=batch_size,
        )
        print(f"{name} clustered and saved to {os.path.join(cluster_dirs[name.lower()], 'clusters.csv')}")


def update_sentences_with_clusterized(
        df: pd.DataFrame, 
        output_dir: str
        ) -> pd.DataFrame:
    '''
    Reads a DataFrame CSV containing phrase-to-label mappings and replaces
    matching phrases in the 'parsed_sentence' column with their cluster labels.
    
    Args:
        df (pd.DataFrame): DataFrame with original phrases
        output_dir (str): Directory where cluster labels are stored
    
    Returns:
        pd.DataFrame: a Pandas DataFrame with phrases replaced by cluster labels in 'parsed_sentence'.
    '''
    meta = {"noun_phrases": 'Noun phrases', "verbs": 'Verbs'}
    print(f"\n{'='*60}")
    for file_name, phrase_type in meta.items():
        print(f"Updating {phrase_type} with clusterized labels...")
        clusters_path = os.path.join(output_dir, file_name, 'clusters.csv')
        clusters_df = pd.read_csv(clusters_path, converters={'phrases': ast.literal_eval})
        phrase2label = {
            phr.lower(): tup.label 
            for tup in clusters_df.itertuples() 
            for phr in tup.phrases 
        }
        df = df.copy()
        df["parsed_sentence"] = df["parsed_sentence"].apply(
            lambda lst: [phrase2label.get(w.lower(), w) for w in lst]
        )
        print(f"{phrase_type} updated with clusters and saved to {clusters_path}")
    print(f"\n{'='*60}")
    return df
