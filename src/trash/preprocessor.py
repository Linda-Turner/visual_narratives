"""
Based on Embeddings class from relatio package: https://github.com/relatio-nlp/relatio/blob/master/relatio/preprocessing.py
Original copyright (c) 2023-2024 ETH Zurich, Andrei V. Plamada, et al.
Modified by Dariia Puhach, Info-Lab, Uppsala University, 2025.
Licensed under MIT License.
"""
import csv
import time

import pandas as pd
import spacy
from spacy.cli import download as spacy_download
from tqdm import tqdm

class Preprocessor:
    """
    A class to preprocess a given corpus (e.g., split it into sentences, clean)

    Args:
        spacy_model: One of the available spacy models for the English language (default: en_core_web_sm). For a complete list, see: https://spacy.io/models/en
        remove_punctuation: whether to remove string.punctuation
        remove_digits: whether to remove string.digits
        stop_words: list of stopwords to remove
        lowercase: whether to lower the case
        lemmatize: whether to lemmatize
        n_process: Number of processes to user in nlp.pipe() for parallel computing (default: -1). Set to -1 to use all cores on the machine.
        batch_size: Size of the batches for parallel computing (default: 1000 -- the SpaCy default).

    Note:
        self.nlp.add_pipe("sentencizer") => rule-based sentence segmentation without the dependency parse.
    """
    remove_chars = ["\"",'-',"^",".","?","!",";","(",")",",",":","\'","+","&","|","/","{","}",
                "~","_","`","[","]",">","<","=","*","%","$","@","#","’"]

    def __init__(
        self,
        spacy_model="en_core_web_sm",
        remove_punctuation: bool = True,
        remove_digits: bool = True,
        stop_words: list = [],
        lowercase: bool = True,
        lemmatize: bool = True,
        remove_chars: list = remove_chars,
        n_process: int = 1,
        # batch_size: int = 1000,
        coref_batch_size: int = 64,
        sentencizer_batch_size: int = 1000,
    ):
        if not spacy.util.is_package(spacy_model):
            spacy_download(spacy_model)

        self.spacy_model = spacy_model
        self.coref_nlp = spacy.load(spacy_model)
        self.sentencizer_nlp = spacy.load(spacy_model)
        self.sentencizer_nlp.add_pipe("sentencizer", first=True)
        self.n_process = n_process
        self.coref_batch_size = coref_batch_size
        self.sentencizer_batch_size = sentencizer_batch_size
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.remove_chars = remove_chars

    def resolve_coreferences(self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Batch process multiple texts for primitive pronoun resolution using spaCy's pipe.
        
        Args:
            df: Pandas DataFrame with a column "Labels" containing the texts to process.
            
        Returns:
            Pandas DataFrame with the coreference-resolved texts updated in the "Labels" column.
        """
        print(f"\n{'='*60}")
        print("Coreference resolution...")
        
        spacy_docs = self.coref_nlp.pipe(df["Labels"], batch_size=self.coref_batch_size, n_process=self.n_process)
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

    def split_into_sentences(
        self,
        df: pd.DataFrame,
        output_path: str = None,
    ) -> pd.DataFrame:
        
        """
        Batch process multiple texts for splitting sentences using spaCy's sentence splitter.

        Args:
            dataframe: a Pandas DataFrame with a column "ImageID" and a column "Labels"
            output_path: path to save the Pandas DataFrame in a .csv format (default is None).
            NB: output_path is worth specifying for large datasets, since in this case, the result
            is written to a file row by row and then read back into a DataFrame.

        Returns:
            Pandas DataFrame with all the columns which were in the input dataframe

        """
        spacy_docs = self.sentencizer_nlp.pipe(
            df["Labels"],
            disable=["tagger", "ner", "parser", "lemmatizer"],
            batch_size=self.sentencizer_batch_size,
            n_process=self.n_process,
        )

        print(f"\n{'='*60}")
        print("Splitting into sentences...")
        time.sleep(1)

        if output_path is None:
            raise NotImplementedError("For large datasets, please specify output_path")

        with open(output_path, "w", encoding="utf-8",newline="") as csvfile:
            fieldnames = list(df.columns)
            fieldnames.append('sentence')
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            # for i, doc in enumerate(tqdm(spacy_docs, total=len(df))):
            #     curr_row = df.iloc[i].tolist()
            #     for sent in doc.sents:
            #         sentence = str(sent)
            #         writer.writerow([*curr_row, sentence])
            for row, doc in tqdm(zip(df.itertuples(index=False), spacy_docs), total=len(df)):
                curr_row = list(row)
                for sent in doc.sents:
                    sentence = str(sent)
                    # filter out two-symbol sentences
                    if len(sentence) > 2:
                        writer.writerow([*curr_row, sentence])

        sentence_df = pd.read_csv(output_path)
        sentence_df = sentence_df.drop(columns=['Labels'])

        return sentence_df