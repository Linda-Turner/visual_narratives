import os

import pandas as pd

from preprocessing.Sentence_Splitter import SentenceSplitter
from trash.Coreference_Resolution import CoreferenceResolver
from preprocessing.Sentence_Parser import SentenceParser

class NarrativePreprocessor:
    """
    A class to manage the preprocessing pipeline for visual narratives.

    1. Filter out invalid descriptions (i.e. where the model failed to provide one)
    2. Resolve coreferences
    3. Split each description into sentences
    4. Parse each sentence with a syntactic parser (here: benepar)

    Args:
        n_process: Number of processes to user in nlp.pipe() for parallel computing (default: 1). Set to -1 to use all cores on the machine.
        coref_batch_size: Size of the batches for coreference resolution (default: 64).
        splitter_batch_size: Size of the batches for sentence splitting (default: 1000).
        parser_batch_size: Size of the batches for sentence parsing (default: 1000).
        spacy_model: spaCy model to use for dependency parsing and sentence splitting (default: "en_core_web_sm").
        benepar_model: Benepar parsing model (default: "benepar_en3").
    """
    def __init__(self, 
        n_process: int = 1, 
        coref_batch_size=64,
        splitter_batch_size=1000,
        parser_batch_size=1000,
        spacy_model: str = "en_core_web_sm", 
        benepar_model: str = "benepar_en3"
        ):
        self.coref = CoreferenceResolver(n_process=n_process, batch_size=coref_batch_size, spacy_model=spacy_model)
        self.splitter = SentenceSplitter(n_process=n_process, batch_size=splitter_batch_size, spacy_model=spacy_model)
        self.parser = SentenceParser(n_process=n_process, batch_size=parser_batch_size, spacy_model=spacy_model, benepar_model=benepar_model)

    def preprocess(self, df: pd.DataFrame, output_dir: str) -> str:
        """
        Run the complete preprocessing pipeline.

        Args:
            df: .tsv file with columns: 'Dir', 'ImageID', 'Labels'. 
                'Labels' contain step 2 descriptions generated in earlier stages in the pipeline.
            output_dir: Directory where intermediate and final processed files
                will be saved.

        Returns:
            Path to the parsed sentences CSV file.
        """
        df = self.filter_descriptions(df)

        # The resolution is quite primitive but resolves 94.889% of problems
        df = self.coref.resolve_coreferences(df)

        output_path_sentences = os.path.join(output_dir, 'sentences.csv')
        self.splitter.split_into_sentences(
            df,
            output_path=output_path_sentences
        )

        output_path_parsed = os.path.join(output_dir, 'parsed_sentences.csv')
        self.parser.parse_sentences(
            pd.read_csv(output_path_sentences),
            output_path=output_path_parsed
        )
        return output_path_parsed

    def filter_descriptions(self, df):
        """
        Filter out invalid descriptions from the DataFrame. (i.e. where the model failed to provide a description)

        Args:
            df: Pandas DataFrame with a column "Labels" containing the descriptions.

        Returns:
            Filtered Pandas DataFrame.
        """
        invalid_patterns = [
            "clear narrative",
            "no narrative",
            "I can't",
            "image does not",
        ]

        for pattern in invalid_patterns:
            df = df[
                ~df["Labels"].str.contains(pattern, case=False)
            ]

        return df