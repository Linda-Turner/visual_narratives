import re
import csv

import pandas as pd
import benepar
import spacy
from nltk import Tree
from tqdm import tqdm
from spacy.cli import download as spacy_download

from .orchestrator import SPACY_MODEL, BENEPAR_MODEL

class SentenceParser:
    """
    A class to parse sentences into syntactic components using Benepar.

    Args:
        spacy_model (str): spaCy model to use for dependency parsing and sentence splitting Default to "en_core_web_sm". 
            For a complete list, see: https://spacy.io/models/en
        benepar_model (str): Benepar parsing model Default to "benepar_en3".
        n_process (int): Number of processes to user in nlp.pipe() for parallel computing Default to 1. 
            Set to -1 to use all cores on the machine.
        batch_size (int): Size of the batches for parallel computing Default to 1000.

    Note:
        Benepar adds a constituency parser to spaCy, producing syntactic trees that are converted into labeled sentence components.
    """

    def __init__(
        self,
        spacy_model=SPACY_MODEL,
        benepar_model=BENEPAR_MODEL,
        n_process: int = 1,
        batch_size: int = 1000,
    ):

        if not spacy.util.is_package(spacy_model):
            spacy_download(spacy_model)
        try:
            benepar.load(benepar_model)
        except:
            benepar.download(benepar_model)
        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)
        self.nlp.add_pipe(
            "benepar",
            config={"model": benepar_model}
        )
        # extra nlp model for lemmatization, without parser and ner
        self.lemma_nlp = spacy.load(
            spacy_model,
            disable=["parser", "ner"]
        )
        self.n_process = n_process
        self.batch_size = batch_size

    def parse_sentences(
            self,
            df: pd.DataFrame,
            output_path: str,
        ) -> None:
        """
        Parse the sentences in the 'sentence' column of a Pandas DataFrame and save the result to a CSV file.

        Args:
            df (pd.DataFrame): a Pandas DataFrame with the column "sentence" containing the sentences to parse.
            output_path (str): Path to save the parsed results in a .csv format
        """
        print(f"\n{'='*60}")
        print("Parsing sentences...")

        spacy_docs = self.nlp.pipe(df["sentence"], batch_size=self.batch_size, n_process=self.n_process)

        # write to csv in order not to overload the memory
        if output_path is None:
            raise NotImplementedError("Please specify output_path")
        
        with open(output_path, 'w', encoding="utf-8",newline='') as csvfile:
            fieldnames = ['sentence_id', 'parsed_labels', 'parsed_sentence'] + df.columns.tolist()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row, doc in tqdm(zip(df.itertuples(), spacy_docs), total=len(df)):
                labels, sentence = self.parse_doc(doc)
                row_data = row._asdict()
                sentence_id = row_data.pop("Index")
                writer.writerow({
                    "sentence_id": sentence_id,
                    "parsed_labels": labels,
                    "parsed_sentence": sentence,
                    **row_data,
                })

    def parse_doc(
            self, 
            doc: spacy.tokens.Doc
        ) -> tuple[list, list]:
        '''
        Wrapper function to parse the doc of the sentence into its syntactic components using Benepar.

        Args:
            doc (spacy.tokens.Doc): A spaCy Doc object representing the sentence to parse.

        Returns:
            tuple[list,list]: list of syntactic labels and list of corresponding sentence pieces
        '''
        sents = list(doc.sents)
        if not sents:
            return [], []
        
        tree = Tree.fromstring(sents[0]._.parse_string)
        parsed = traverse_tree(tree, [])
        parsed_labels, parsed_sentence = self.clean_parsed(parsed)
        return parsed_labels, parsed_sentence
        

    def clean_parsed(
            self, 
            parsed: list[tuple]
        ) -> tuple[list, list]:
        '''
        Clean the punctuation from the parsed list and lemmatize verbs.
        Keeps gerunds.

        Args:
            parsed (list[tuple]): A list of tuples containing syntactic labels and corresponding sentence pieces.

        Returns:
            tuple[list,list]: a tuple containing a list of syntactic labels and list of corresponding sentence pieces
        '''
        parsed_sentence = []
        parsed_labels = []
        for label, s_piece in parsed:
            if not label.isalpha() or label == 'HYPH':
                continue
            parsed_labels.append(label)
            
            # Lemmatize verbs but keep gerunds
            if label.startswith('VB') and label != 'VBG':
                spacy_docs = self.lemma_nlp(s_piece)
                lemmatized = ' '.join([token.lemma_ for token in spacy_docs])
                parsed_sentence.append(lemmatized)
            else:
                parsed_sentence.append(s_piece)
        
        return parsed_labels, parsed_sentence


def traverse_tree(
        tree: Tree, 
        parsed: list
    ) -> list:
    '''
    Recursive function to traverse the syntactic tree and extract components.
    If the branch is a pure noun phrase, like "ice cover variations", it will be non chopped.
    Otherwise, it will be parsed into smaller pieces.

    TODO: parse CC into two different sentences.
    'The image shows street signs and a placard warning' =>
      'The image shows street signs'
      'The image shows a placard warning'

    Args:
        tree (Tree): A nltk Tree object representing the syntactic structure of a sentence.
        parsed (list): A list to accumulate the parsed components.
    
    Returns:
        list: a parsed list of tuples containing syntactic labels and corresponding sentence piece. e.g. [('NP', 'Presentation screen'), ('VBZ', 'displays'), ('NP', 'the logo')].
    '''
    for branch in tree:
        if branch.label() == "NP" and _contains_nested(branch):
            traverse_tree(branch, parsed)
        elif branch.label() in ("VP", "PP", "S", "SBAR", "ADJP"):
            traverse_tree(branch, parsed)
        else:
          joined_leaves = ' '.join(branch.leaves())
          parsed.append((branch.label(), joined_leaves))
    return parsed


def _contains_nested(
        np_branch: Tree
    ) -> bool:
    """
    Check if an NP contains nested structures:
        PP (prepositional clause) or SBAR (semantic dependent clause),
        S (dependent clause), or VP (verb phrase).
    
    Args:
        np_branch (Tree): A nltk Tree object representing a noun phrase (NP) branch.

    Returns:
        bool: True if the NP contains nested structures, False otherwise.
    """
    for leaf in np_branch:
        if leaf.label() in ("PP", "SBAR", "S", "VP"):
            return True
    return False
  


