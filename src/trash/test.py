import ast
from collections import defaultdict
import csv
import os

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

def create_clusters(input_file: str, output_dir: str, batch_size: int = 15000, pca_args: dict = {'n_components': 50, 'svd_solver': 'full'}):
    """
    Creates phrase clusters.
    If the dataset is larger than batchsize, process phrases in batches to handle memory constraints, 
    creates initial clusters within batches, then performs second-level clustering
    on batch labels.
    
    Args:
        input_file: Path to input CSV file containing a 'word' column.
        output_dir: Directory path where output files will be saved.
        batch_size: Maximum number of phrases per batch. Defaults to 15000.
        pca_args: Dictionary of PCA parameters. Defaults to 50 components
            with 'full' SVD solver.
    
    """
    batch_clusters_path = os.path.join(output_dir, 'batch_clusters.csv')
    clusters_path = os.path.join(output_dir, 'clusters.csv')

    df = pd.read_csv(input_file)
    df['count'] = df.groupby('word')['word'].transform('count')
    phrase2count = df.drop_duplicates('word').set_index('word')['count'].to_dict()
    phrases = df['word'].unique().tolist()

    # Simple clustering
    if len(phrases) <= batch_size:
        with open(clusters_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'phrases', 'size'])    
            write_clusters(phrases, phrase2count, writer, pca_args)
        return
    # Batch clustering
    with open(batch_clusters_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'phrases', 'size'])
        
        for i in range(0, len(phrases), batch_size):
            batch = phrases[i:i+batch_size]
            print(f'Embedding batch {i//batch_size + 1} of {(len(phrases)-1)//batch_size + 1}')
            write_clusters(batch, phrase2count, writer, pca_args)

    df = pd.read_csv(batch_clusters_path, converters={'phrases': ast.literal_eval})
    label2phrases = {row.label: row.phrases for row in df.itertuples(index=False)}
    
    clusters = embed_and_cluster(df['label'].tolist(), pca_args) 
    with open(clusters_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'phrases', 'size'])
        for cluster in tqdm(clusters):
            df_subset = df.iloc[cluster]
            label2size = df_subset.groupby('label')['size'].sum().to_dict()
            label = max(label2size, key=label2size.get)
            size = sum(label2size.values())
            clustered_phrases = [
                phrase 
                for lab in df_subset['label']
                for phrase in label2phrases[lab]
            ]
            
            writer.writerow([label, clustered_phrases, size])    


def write_clusters(phrases: list[str],
    phrase2count: dict[str, int],
    writer: csv.writer,
    pca_args: dict,):

    batch_clusters = embed_and_cluster(phrases, pca_args)
    id2phrase = {idx: phrase for idx, phrase in enumerate(phrases)}

    for cl in tqdm(batch_clusters): 
        cl_phrases = [id2phrase[idx] for idx in cl] 
        counts = [phrase2count[ph] for ph in cl_phrases] 
        cl_label = cl_phrases[counts.index(max(counts))] 
        cl_size = sum(counts)
        
        writer.writerow([cl_label, cl_phrases, cl_size])






def create_clusters_batched(input_file: str, output_dir: str, batch_size: int = 15000, pca_args: dict = {'n_components': 50, 'svd_solver': 'full'}):
    """
    Creates phrase clusters in batches for large datasets.

    Calls create_clusters() if dataset fits in one batch.
    
    Processes phrases in batches to handle memory constraints, creates
    initial clusters within batches, then performs second-level clustering
    on batch labels.
    
    Args:
        input_file: Path to input CSV file containing a 'word' column.
        output_dir: Directory path where output files will be saved.
        batch_size: Maximum number of phrases per batch. Defaults to 15000.
        pca_args: Dictionary of PCA parameters. Defaults to 50 components
            with 'full' SVD solver.
    
    """
    batch_clusters_path = os.path.join(output_dir, 'batch_clusters.csv')
    clusters_path = os.path.join(output_dir, 'clusters.csv')

    df = pd.read_csv(input_file)
    df['count'] = df.groupby('word')['word'].transform('count')
    phrase2count = df.drop_duplicates('word').set_index('word')['count'].to_dict()
    phrases = df['word'].unique().tolist()
    if len(phrases) <= batch_size:
        return create_clusters(input_file, output_dir, pca_args)

    with open(batch_clusters_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'phrases', 'size'])
        
        for i in range(0, len(phrases), batch_size):
            batch = phrases[i:i+batch_size]
            print(f'Embedding batch {i//batch_size + 1} of {(len(phrases)-1)//batch_size + 1}')
        
            batch_clusters = embed_and_cluster(batch, pca_args)
            id2phrase = {idx: phrase for idx, phrase in enumerate(batch)}

            for cl in tqdm(batch_clusters): 
                cl_phrases = [id2phrase[idx] for idx in cl] 
                counts = [phrase2count[ph] for ph in cl_phrases] 
                cl_label = cl_phrases[counts.index(max(counts))] 
                cl_size = sum(counts)
                
                writer.writerow([
                    cl_label, cl_phrases, cl_size,
                ])

    df = pd.read_csv(batch_clusters_path, converters={'phrases': ast.literal_eval})
    label2phrases = {row.label: row.phrases for row in df.itertuples(index=False)}
    
    clusters = embed_and_cluster(df['label'].tolist(), pca_args) 
    with open(clusters_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'phrases', 'size'])
        for cluster in tqdm(clusters):
            df_subset = df.iloc[cluster]
            label2size = df_subset.groupby('label')['size'].sum().to_dict()
            label = max(label2size, key=label2size.get)
            size = sum(label2size.values())
            clustered_phrases = [
                phrase 
                for lab in df_subset['label']
                for phrase in label2phrases[lab]
            ]
            
            writer.writerow([label, clustered_phrases, size])

def create_clusters(
    input_path: str, save_folder_path: str, 
    pca_args: dict = {'n_components': 50, 'svd_solver': 'full'}
):
    """
    Creates phrase clusters from a CSV file and save results.
    
    Reads phrases from a CSV with a 'word' column, groups them into clusters
    using embeddings and PCA, then saves clusters with their most frequent
    phrase as the label.
    
    Args:
        input_path: Path to input CSV file containing a 'word' column.
        save_folder_path: Directory path where clusters.csv will be saved.
        pca_args: Dictionary of PCA parameters. Defaults to 50 components
            with 'full' SVD solver.
    """
    os.makedirs(save_folder_path, exist_ok=True)

    df = pd.read_csv(input_path)
    df['count'] = df.groupby('word')['word'].transform('count')
    phrase2count = df.drop_duplicates('word').set_index('word')['count'].to_dict()
    phrases = df['word'].unique().tolist()
    clusters = embed_and_cluster(phrases, pca_args)
    id2phrase = {idx: phrase for idx, phrase in enumerate(phrases)}
    
    with open(os.path.join(save_folder_path, 'clusters.csv'), 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'phrases', 'size'])
        for cluster in tqdm(clusters):
            phrases = [id2phrase[idx] for idx in cluster]
            counts = [phrase2count[ph] for ph in phrases]
            label = phrases[counts.index(max(counts))]
            size = sum(counts)
            writer.writerow([label, phrases, size])