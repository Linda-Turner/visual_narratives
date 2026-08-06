import re
import csv

import pandas as pd
import benepar
import spacy
from nltk import Tree
from tqdm import tqdm

benepar.download('benepar_en3')

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})


def traverse_tree(tree: Tree, parsed: list) -> list:
    '''
    If the branch is a pure noun phrase, like "ice cover variations", it will be non chopped.
    Otherwise, it will be prased into smaller pieces.

    TODO: parse CC into two different sentences.
    'The image shows street signs and a placard warning' =>
      'The image shows street signs'
      'The image shows a placard warning'
    
    Returns:
        a parsed list of tuples, like:
        [('NP', 'Presentation screen'), ('VBZ', 'displays'), ('NP', 'the logo')].
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


def _contains_nested(np_branch) -> bool:
    """
    Check if an NP contains nested structures:
        PP (prepositional clause) or SBAR (semantic dependent clause),
        S (dependent clause), or VP (verb phrase).
    """
    for leaf in np_branch:
        if leaf.label() in ("PP", "SBAR", "S", "VP"):
            return True
    return False
  

def clean_parsed(parsed: list[tuple]) -> tuple[list, list]:
    '''
    Clean the punctuation from the parsed list and lemmatize verbs.
    Keeps gerunds.
    '''
    parsed_sentence = []
    parsed_labels = []
    for label, piece in parsed:
        if not label.isalpha() or label == 'HYPH':
            continue
        parsed_labels.append(label)
        
        # Lemmatize verbs but keep gerunds
        if label.startswith('VB') and label != 'VBG':
            doc = nlp(piece)
            lemmatized = ' '.join([token.lemma_ for token in doc])
            parsed_sentence.append(lemmatized)
        else:
            parsed_sentence.append(piece)
    
    return parsed_labels, parsed_sentence

def parse_sentence(
        df: pd.DataFrame,
        output_path: str = None,
    ):# -> tuple[list, list]:
    '''
    Parse the sentence into its syntactic components using Benepar.
    Returns:
        parsed_labels: list of syntactic labels
        parsed_sentence: list of corresponding sentence pieces
    '''
    if output_path is None:
        raise NotImplementedError("For large datasets, please specify output_path")
    # write to csv in order not to overload the memory
    with open(output_path, 'w',encoding="utf-8", newline='') as csvfile:
        fieldnames = ['sentence_id', 'parsed_labels', 'parsed_sentence'] + df.columns.tolist()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        tqdm.write("Parsing sentences...")
        # for idx, row in df_sentences.iterrows():
        for idx, row in tqdm(df.iterrows(),total=len(df)):
            sentence = row['sentence']
            sentence = re.sub(r"\s+", " ", sentence).strip()
            doc = nlp(sentence)
            sents = list(doc.sents)
            if not sents:
                return [], []

            tree = Tree.fromstring(sents[0]._.parse_string)
            parsed = traverse_tree(tree, [])
            labels, sentence = clean_parsed(parsed)
            # labels, sentence = parse_sentence(row['sentence'])
            row_data = row.to_dict()
            writer.writerow({
                'sentence_id': idx,
                'parsed_labels': labels,
                'parsed_sentence': sentence,
                **row_data,
            })

    #return parsed_labels, parsed_sentence
