import spacy
from tqdm import tqdm
import pandas as pd

from spacy.cli import download as spacy_download

def resolve_coreferences(
        df: pd.DataFrame, 
        n_process: int = 1, 
        batch_size: int = 64, 
        spacy_model: str = "en_core_web_sm"
        ) -> pd.DataFrame:
    """
    Replace simple subject pronouns with the first subject in each description.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Labels' column with text descriptions.
        n_process (int): Number of processes to user in nlp.pipe() for parallel computing (default: 1). Set to -1 to use all cores on the machine.
        batch_size (int): Size of the batches for parallel computing (default: 1000 -- the SpaCy default).
        spacy_model (str): spaCy model to use for dependency parsing and sentence splitting (default: "en_core_web_sm"). 
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
