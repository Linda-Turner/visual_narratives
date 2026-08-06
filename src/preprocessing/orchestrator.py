import os

import pandas as pd

from preprocessing.coref_resolution import resolve_coreferences
from preprocessing.sentence_splitter import SentenceSplitter
from preprocessing.sentence_parser import SentenceParser

def filter_descriptions(df : pd.DataFrame) -> pd.DataFrame:
    """
    Filter out invalid descriptions from the DataFrame. (i.e. where the model failed to provide a description)

    Args:
        df (pd.DataFrame): Pandas DataFrame with a column "Labels" containing the descriptions.

    Returns:
        pd.DataFrame: a Pandas DataFrame with filtered descriptions.
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

def preprocess_sentences(
        input_path: str, 
        output_dir: str, 
        n_process: int=1, 
        coref_batch_size: int=64, 
        splitter_batch_size: int=1000, 
        parser_batch_size: int=1000, 
        spacy_model: str="en_core_web_sm", 
        benepar_model: str="benepar_en3"
        ) -> None:
    '''
    A wrapper function to preprocess the sentences from the file in the input path and save the processed sentences to the output directory.
    The preprocessing steps include:
        1. Filter out invalid descriptions (i.e. where the model failed to provide one)
        2. Resolve coreferences
        3. Split each description into sentences
        4. Parse each sentence with a syntactic parser (here: benepar)

    Args: 
        input_path (str): .tsv file with a 'Labels' column with text descriptions. 
        output_dir (str): directory to save the processed sentences.
        n_process (int): Number of processes to user in nlp.pipe() for parallel computing (default: 1). 
            Set to -1 to use all cores on the machine.
        coref_batch_size (int): Size of the batches for coreference resolution (default: 64).
        splitter_batch_size (int): Size of the batches for sentence splitting (default: 1000).
        parser_batch_size (int): Size of the batches for sentence parsing (default: 1000).
        spacy_model (str): spaCy model to use for dependency parsing and sentence splitting (default: "en_core_web_sm"). 
            For a complete list, see: https://spacy.io/models/en
        benepar_model (str): Benepar parsing model (default: "benepar_en3").

    Returns:
        str: a path to the preprocessed parsed sentences CSV file.
    '''
    splitter = SentenceSplitter(n_process=n_process, batch_size=splitter_batch_size, spacy_model=spacy_model)
    parser = SentenceParser(n_process=n_process, batch_size=parser_batch_size, spacy_model=spacy_model, benepar_model=benepar_model)

    df = pd.read_csv(input_path, sep='\t')

    # 1. 
    df = filter_descriptions(df)

    # 2. 
    df = resolve_coreferences(df, n_process=n_process, batch_size=coref_batch_size, spacy_model=spacy_model)

    # 3. 
    output_path_sentences = os.path.join(output_dir,'sentences.csv')
    splitter.split_into_sentences(
        df,
        output_path=output_path_sentences
    )

    # 4. 
    output_path_parsed = os.path.join(output_dir ,'parsed_sentences.csv')
    parser.parse_sentences(
        pd.read_csv(output_path_sentences),
        output_path=output_path_parsed
    )

    return output_path_parsed