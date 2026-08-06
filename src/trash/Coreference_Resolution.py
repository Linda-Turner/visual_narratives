import pandas as pd
import spacy
from spacy.cli import download as spacy_download
from tqdm import tqdm

class CoreferenceResolver:
    """
    A class for resolving primitive coreference expressions in text descriptions.

    This implementation uses spaCy dependency parsing to replace simple pronouns
    (i.e., "he", "she", "they", "it") with the first detected subject in the
    sentence. 

    Args:
        spacy_model: One of the available spacy models for the English language (default: en_core_web_sm). Default: "en_core_web_sm". For a complete list, see: https://spacy.io/models/en
        spacy_model: spaCy model to use for dependency parsing. 
        n_process: Number of processes to user in nlp.pipe() for parallel computing (default: 1). Set to -1 to use all cores on the machine.
        batch_size: Size of the batches for parallel computing (default: 1000 -- the SpaCy default).
    """
    def __init__(
        self,
        spacy_model="en_core_web_sm",
        n_process: int = 1,
        batch_size: int = 1000,
    ):
        if not spacy.util.is_package(spacy_model):
            spacy_download(spacy_model)
        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)
        self.n_process = n_process
        self.batch_size = batch_size

    def resolve_coreferences(self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply heuristic coreference resolution to descriptions.

        Args:
            df: DataFrame containing a 'Labels' column with text descriptions.

        Returns:
            The input DataFrame with the 'Labels' column updated.
        """
        print(f"\n{'='*60}")
        print("Coreference resolution...")
        
        spacy_docs = self.nlp.pipe(df["Labels"], batch_size=self.batch_size, n_process=self.n_process)
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


