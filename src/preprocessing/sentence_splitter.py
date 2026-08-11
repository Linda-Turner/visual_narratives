"""
Based on Embeddings class from relatio package: https://github.com/relatio-nlp/relatio/blob/master/relatio/preprocessing.py
Original copyright (c) 2023-2024 ETH Zurich, Andrei V. Plamada, et al.
Modified by Dariia Puhach, Info-Lab, Uppsala University, 2025.
Licensed under MIT License.
"""
import csv

import pandas as pd
import spacy
from spacy.cli import download as spacy_download
from tqdm import tqdm

from preprocessing.config import SPACY_MODEL

class SentenceSplitter:
    """
    A class to split text descriptions into sentences using spaCy.

    Args:
        spacy_model (str): One of the available spacy models for the English language Default to en_core_web_sm. 
            For a complete list, see: https://spacy.io/models/en
        n_process (int): Number of processes to user in nlp.pipe() for parallel computing Default to 1. 
            Set to -1 to use all cores on the machine.
        batch_size (int): Size of the batches for parallel computing Default to 1000.

    Note:
        Uses spaCy's rule-based sentencizer, which splits sentences without requiring dependency parsing.
    """

    def __init__(
        self,
        spacy_model=SPACY_MODEL,
        n_process: int = 1,
        batch_size: int = 1000,
    ):
        if not spacy.util.is_package(spacy_model):
            spacy_download(spacy_model)

        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)
        self.nlp.add_pipe("sentencizer", first=True)
        self.n_process = n_process
        self.batch_size = batch_size

    def split_into_sentences(
        self,
        df: pd.DataFrame,
        output_path: str,
    ) -> None:
        """
        Split sentences in the 'Labels' column of a DataFrame and save the result to a CSV file.

        Args:
            df (pd.DataFrame): a Pandas DataFrame with the column "Labels" containing the text descriptions to split into sentences.
            output_path (str): path to save the split sentences in a .csv format.
                Note: The result is written to a file row by row.
        """
        spacy_docs = self.nlp.pipe(
            df["Labels"],
            disable=["tagger", "ner", "parser", "lemmatizer"],
            batch_size=self.batch_size,
            n_process=self.n_process,
        )

        print(f"\n{'='*60}")
        print("Splitting into sentences...")

        # write to csv in order not to overload the memory
        if output_path is None:
            raise NotImplementedError("Please specify output_path")

        # The original descriptions are no longer needed after sentence splitting, so remove them before writing to reduce output size.
        df = df.copy()
        df = df.drop(columns=['Labels'])

        with open(output_path, "w", encoding="utf-8",newline="") as csvfile:
            fieldnames = list(df.columns)
            fieldnames.append('sentence')
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for row, doc in tqdm(zip(df.itertuples(index=False), spacy_docs), total=len(df)):
                curr_row = list(row)
                for sent in doc.sents:
                    sentence = str(sent)
                    # filter out two-symbol sentences
                    if len(sentence) > 2:
                        writer.writerow([*curr_row, sentence])