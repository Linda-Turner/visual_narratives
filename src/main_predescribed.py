import ast
import argparse
import os

import pandas as pd

from generate_description.orchestrator import two_step_descriptions
from preprocessing.orchestrator import preprocess_sentences
from clustering.orchestrator import (cluster_verb_and_noun_phrases, update_sentences_with_clusterized)
from make_graphs.build_graph import create_and_save_multiple_graphs
from make_graphs.draw_graph import draw_graph

placeholder = None
PROMPT1 = f"I am a researcher who researches the narratives around climate change. You are a climate change {placeholder}. First, answer in one sentence: With what purpose do you use the images of famous symbols, people (like Greta Thunberg), or places in the image? (If there are not any famous symbols, people, or places, just skip that.) Then, describe the image. Afterwards, think carefully and tell me what narrative you would want to convey, given that you posted this image together with the text on Twitter."
PROMPT2 = "Based on your response above, reformat your answer into the following format. Use only the part of the response that describes a narrative, do not focus on everything above it. Write 1-3 simple SVO sentences in the structure “who does what to whom.” For example, “Wasting money is bad for economy. Wasting money reduces savings.” Please preserve all proper nouns. Do not use pronouns, instead reuse the same words. Write a message as if you wrote a tweet."

def build_visual_narratives(input_file: str, output_dir: str, generation, description_input_file:str):
    '''
    The complete visual narrative generation pipeline from step 2 descriptions, 
    including preprocessing, syntactic parsing, phrase clustering, and graph construction.

    Args:
        Input_file (str): .tsv file with columns: 'Dir', 'ImageID', 'Labels'. 
            'Labels' contain step 2 descriptions generated in earlier stages in the pipeline.
        Output_dir (str): directory to save the narratives
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    data_output_dir = os.path.join(output_dir, "data")
    if not os.path.exists(data_output_dir):
        os.makedirs(data_output_dir)
    
    # Describe images
    if generation:
        df = pd.read_csv(input_file, sep="\t")
        image_dir = os.path.commonpath(df["path"].tolist())
        description_step1_file = os.path.join(data_output_dir,'descriptions_step1.tsv')
        description_step2_file = os.path.join(data_output_dir,'descriptions_step2.tsv')
        two_step_descriptions(input_df=df, model='gemma', image_dir=image_dir, 
                             outfile_step1=description_step1_file, outfile_step2=description_step2_file,
                             prompt_step1=PROMPT1,prompt_step2=PROMPT2, credentials=None,
                             max_tokens=2000, limit=-1, split=None)
        description_input_file = description_step2_file


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visual narratives from an input TSV file.")
    parser.add_argument("-i", "--input_file",
        help="Path to the input TSV file containing at least the columns 'event', 'usr_type', and 'path', describing the images to classify." \
            "'path' contains the directory path to the images to classify", 
        required=True)
    parser.add_argument("-d", "--output-dir",
        help="Directory where generated files will be saved.",
        default="output_narratives")
    parser.add_argument("-g", "--generation",
        help="To generate image descriptions or not.",
        type=lambda x: x.lower() == "true",
        default=True)
    parser.add_argument("-di", "--description_input_file",
                        help="Path to the description input TSV file containing at least the columns 'Dir', 'ImageID', and 'Labels', containing the descriptions of the images.", default=None)
    parser.add_argument('-m','--model',
        help='model: gpt4 (default), llava',
        default='gpt4')
    parser.add_argument('-p','--prompt',
        help='prompt to be used for the annotation',
        default='Describe this image')
    parser.add_argument('-c','--credentials',
        help='file with openai credentials',
        default='openai_api.txt')
    parser.add_argument('-t','--tokens',
        help='number of tokens in description',
        default=2000,
        type = int)
    parser.add_argument('-u','--limit',
        help='max number of images to annotate (-1 if no limit)',
        default=-1,
        type = int)
    parser.add_argument('-s','--split',
        help='only annotate files with a numeric filename that is odd (o) or even (e)',
        default='none')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_visual_narratives(args.input_file, args.output_dir, args.generation, args.description_input_file)
    #, args.model, args.prompt, args.credentials, args.tokens, args.limit, args.split
