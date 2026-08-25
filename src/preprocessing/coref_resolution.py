import spacy
from tqdm import tqdm
import pandas as pd

from spacy.cli import download as spacy_download
from preprocessing.config import SPACY_MODEL, COREF_BATCH_SIZE

def resolve_coreferences(
        df: pd.DataFrame, 
        n_process: int, 
        batch_size: int = COREF_BATCH_SIZE, 
        spacy_model: str = SPACY_MODEL
    ) -> pd.DataFrame:
    """
    Replace simple subject pronouns with the first subject in each description.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Labels' column with text descriptions.
        n_process (int): Number of processes to user in nlp.pipe() for parallel computing Defaults to 1. Set to -1 to use all cores on the machine.
        batch_size (int): Size of the batches for parallel computing Defaults to 1000.
        spacy_model (str): spaCy model to use for dependency parsing and sentence splitting Defaults to "en_core_web_sm". 
            For a complete list, see: https://spacy.io/models/en

    Returns:
        pd.DataFrame : a Pandas Dataframe with an updated 'Labels' column.
    """
    print(f"\n{'='*60}")
    print("Coreference resolution...")

    if not spacy.util.is_package(spacy_model):
        spacy_download(spacy_model)
    nlp = spacy.load(spacy_model)
    spacy_docs = nlp.pipe(df["Labels"], batch_size=batch_size, n_process=n_process)

    new_texts = []
    for doc in tqdm(spacy_docs, total=len(df)):
        spans = [doc[tok.left_edge.i: tok.right_edge.i+1] for tok in doc if tok.dep_=="nsubj"]
        text = doc.text
        for span in spans:
            if span.text.lower() in ("they", "it", "she", "he"):
                text = text.replace(span.text, spans[0].text)
        new_texts.append(text)
    # Check if the number of resolved texts matches the original DataFrame length
    if len(new_texts) != len(df):
        raise RuntimeError("Coreference resolution produced an unexpected number of texts.")
    df["Labels"] = new_texts
    return df
