import ast
from collections import defaultdict
import csv
import os

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from .embeddings import Embeddings

EMBEDDING_MODEL = Embeddings(normalize=True)          

def prepare_df_to_clustering(
        df: pd.DataFrame, 
        verb_dir: str, 
        noun_phrases_dir: str
        ) -> tuple[str, str]:
    '''
    Creates CSV files of verbs and noun phrases from the Pandas DataFrame and save the result to a CSV file.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing syntaxically parsed sentences.
        verb_dir (str): Directory path to save verb CSV file.
        noun_phrases_dir (str): Directory path to save noun phrases CSV file.

    Returns:
        tuple[str, str]: a tuple of paths to the created verbs and noun phrases CSV files.
    '''
    print(f"\n{'='*60}")
    print('Preparing data for clustering...')

    verbs_path, noun_phrases_path = os.path.join(verb_dir, 'verbs.csv'), os.path.join(noun_phrases_dir, 'noun_phrases.csv')

    # Nouns and verbs are seperately seperated
    with open(verbs_path, 'w', newline='', encoding='utf-8') as verbs_file, \
         open(noun_phrases_path, 'w', newline='', encoding='utf-8') as np_file:
        
        verbs_writer = csv.DictWriter(verbs_file, fieldnames=['word', 'sentence_id', 'position_idx'])
        np_writer = csv.DictWriter(np_file, fieldnames=['word', 'sentence_id', 'position_idx'])
        
        verbs_writer.writeheader()
        np_writer.writeheader()

        # Loop through all phrases to store verbs and noun phrases
        for row in tqdm(df.itertuples(index=False), total=len(df)):
            labels, sentence, sentence_id = row.parsed_labels, row.parsed_sentence, row.sentence_id

            for idx, (label, part) in enumerate(zip(labels, sentence)):
                if not part:
                    continue

                prepared_row = {
                    'word': part,
                    'sentence_id': sentence_id,
                    'position_idx': idx
                }
                
                if label.startswith('V'):
                    verbs_writer.writerow(prepared_row)

                elif label.startswith('NN') or label == 'NP':
                    np_writer.writerow(prepared_row)
    
    return verbs_path, noun_phrases_path

def clusters_and_write(
        input_path: str, 
        output_dir: str, 
        batch_size: int = 15000, 
        pca_args: dict = {'n_components': 50, 'svd_solver': 'full'}
        ) -> None:
    """
    Creates phrase clusters and save the result to a CSV file.
    If the dataset is larger than batchsize, process phrases in batches to handle memory constraints.
    Creates initial clusters within batches, then performs second-level clustering on batch labels.
    
    Args:
        input_path (str): Path to input CSV file containing a 'word' column.
        output_dir (str): Directory path where output files will be saved.
        batch_size (int): Maximum number of phrases per batch. Defaults to 15000.
        pca_args (dict): Dictionary of PCA parameters. Defaults to 50 components
            with 'full' SVD solver.
    """
    batch_clusters_path = os.path.join(output_dir, 'batch_clusters.csv')
    clusters_path = os.path.join(output_dir, 'clusters.csv')

    # Count occurense of phrases
    df = pd.read_csv(input_path)
    df['count'] = df.groupby('word')['word'].transform('count')
    phrase2count = df.drop_duplicates('word').set_index('word')['count'].to_dict()
    phrases = df['word'].unique().tolist()

    # Simple clustering
    if len(phrases) <= batch_size:
        with open(clusters_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'phrases', 'size'])   
            cluster_batch(phrases, phrase2count, writer, pca_args)
        return
    
    # Batch clustering
    with open(batch_clusters_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'phrases', 'size'])
        
        for i in range(0, len(phrases), batch_size):
            batch = phrases[i:i+batch_size]
            tqdm.write(f'Processing batch {i//batch_size + 1} of {(len(phrases)-1)//batch_size + 1}')
            cluster_batch(batch, phrase2count, writer, pca_args)

    df = pd.read_csv(batch_clusters_path, converters={'phrases': ast.literal_eval})
    label2phrases = {row.label: row.phrases for row in df.itertuples(index=False)}

    tqdm.write(f'Second level clustering on batches ...')
    clusters = embed_and_cluster(df['label'].tolist(), pca_args) 
    with open(clusters_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'phrases', 'size'])
        for cluster in clusters:
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


def cluster_batch(
        phrases: list[str],
        phrase2count: dict[str, int],
        writer,
        pca_args: dict,
        ) -> None:
    """
    Cluster a list of phrases and write the resulting clusters to a CSV writer.

    Args:
        phrases (list[str]): List of unique phrases to cluster.
        phrase2count dict[str, int]: Dictionary mapping each phrase to its occurrence count
            in the original dataset.
        writer: CSV writer object used to write cluster results.
        pca_args (dict): Dictionary of PCA parameters. Defaults to 50 components
            with 'full' SVD solver.
    """
    batch_clusters = embed_and_cluster(phrases, pca_args)
    id2phrase = {idx: phrase for idx, phrase in enumerate(phrases)}

    for cl in batch_clusters: 
        cl_phrases = [id2phrase[idx] for idx in cl] 
        counts = [phrase2count[ph] for ph in cl_phrases] 
        cl_label = cl_phrases[counts.index(max(counts))] 
        cl_size = sum(counts)
        
        writer.writerow([cl_label, cl_phrases, cl_size])
 
def embed_and_cluster(
    phrases: list[str], 
    pca_args: dict = {'n_components': 50, 'svd_solver': 'full'},
    threshold: float = 0.7
    ) -> list[list[int]]:
    '''
    Generates embeddings for input phrases, applies PCA dimensionality reduction,
    and performs agglomerative clustering based on cosine similarity with complete linkage 
    to group semantically similar phrases together.
    
    Args:
        phrases (list[str]): List of text strings to embed and cluster.
        pca_args (dict): Dictionary of parameters to pass to PCA initialization.
            Defaults to {'n_components': 50, 'svd_solver': 'full'}.
        threshold (float): Cosine similarity threshold for clustering (0-1).
            Higher values create tighter clusters. Defaults to 0.7.
    
    Returns:
        list[list[int]]: a list of clusters where each cluster is a list of phrase indices.
            Clusters are sorted by size in descending order.
    '''
    vectors = EMBEDDING_MODEL.get_vectors(phrases, progress_bar=True)

    # PCA cannot have more components than samples/features
    pca_args = pca_args.copy()
    pca_args['n_components'] = min(
        pca_args['n_components'],
        vectors.shape[0])

    pca_model = PCA(**pca_args).fit(vectors)
    training_vectors = pca_model.transform(vectors)
    tqdm.write("Clustering phrases")
    clust = AgglomerativeClustering(
        metric="cosine",
        linkage="complete",
        distance_threshold=1 - threshold,
        n_clusters=None
    )
    labels = clust.fit_predict(training_vectors)
    clusters = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[lbl].append(i)

    clusters = list(clusters.values())
    clusters = sorted(clusters, key=lambda x: -len(x))

    return clusters