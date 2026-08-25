### Extracting narratives from visual data
<img width="618" height="568" alt="traveling_narr_cop_m" src="https://github.com/user-attachments/assets/00ff0540-9f61-4272-a419-746f8ecf756c" />

This repo contains all the code used for interpreting images and extracting narratives. 
The main logic is invoked by the 'main.py' script.


### Usage
The pipeline consists of two stages:

1. Description generation to interpret the images and corresponding text using a VLLM.
2. Narrative extraction to preprocesses the generated descriptions, clusters phrases, and constructs narrative graphs.

The two stages require separate environments because they use different versions of `transformers`.

**1. Create different conda environments for description generation and narrative extraction**
Steps for both environments should be run separately

**1.1 Create the conda environments with Python 3.12**
conda create -n visnarr_descriptions python=3.12 -y
or
conda create -n visnarr_narratives python=3.12 -y

**1.2 Activate it**
conda activate visnarr_descriptions
or
conda activate visnarr_narratives

**1.3 Install all packages with pip in the same env**
python -m pip install -r requirements_descriptions.txt
or
python -m pip install -r requirements_narratives.txt

**1.4 Install spacy model**
python -m spacy download en_core_web_sm

**2. Run code**
Depending on the stage different arguments should be passed.

**2.1 Description generation**
conda activate visnarr_descriptions
python main.py --stage description --input_file <input_file> --output-dir <output_directory> 

| Argument              | Description                                                    | Default             |
| --------------------- | -------------------------------------------------------------- | ------------------- |
| `--stage`             | Pipeline stage. Use `descriptions` for this stage.             | Required            |
| `-i`, `--input_file`  | Input TSV containing image paths and post information.         | Required            |
| `-o`, `--output-dir`  | Directory where generated files are saved.                     | `output_narratives` |
| `-m`, `--model`       | Vision-language model to use. Currently `gemma` or `gpt`.      | `gemma`             |
| `-p1`, `--prompt1`    | Prompt used for the first description step.                    | `''`                |
| `-p2`, `--prompt2`    | Prompt used for the second description step.                   | `''`                |
| `-c`, `--credentials` | File containing OpenAI credentials when using GPT.             | `openai_api.txt`    |
| `-t`, `--max_tokens`  | Maximum number of tokens generated in the descriptions.        | `2000`              |
| `-u`, `--limit`       | Maximum number of images to annotate. Use `-1` for all images. | `-1`                |
| `-s`, `--split`       | Process only odd (`o`) or even (`e`) numbered files.           | `none`              |

**2.2 Narrative extraction**
conda activate visnarr_narratives
python main.py --stage narrative --input_file <input_file> --description_input_file <descriptions_input_file> --output-dir <output_directory>

| Argument                          | Description                                                                                                                    | Default             |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------- |
| `--stage`                         | Pipeline stage. Use `narrative` for this stage.                                                                                | Required            |
| `-i`, `--input_file`              | Original input TSV containing `path`, `event`, and `usr_type`.                                                                 | Required            |
| `-di`, `--description_input_file` | TSV containing `Dir`, `ImageID`, and `Labels`.                                                                                 | `None`              |
| `-o`, `--output-dir`              | Directory where generated files are saved.                                                                                     | `output_narratives` |
| `-n`, `--n_processes`             | Number of processes used by spaCy's `nlp.pipe()`.                                                                              | `1`                 |
|                                   | Use `1` for single-process execution or `-1` to use all available CPU cores.                                                   |                     |    


### Input data
The input data is provided as a .tsv file.

The <input_file> must contain at least the following columns:

| Column     | Description                                                                               |
| ---------- | ----------------------------------------------------------------------------------------- |
| `path`     | Path to the image file                                                                    |
| `text`     | Text associated with the image                                                            |
| `event`    | Specific issue or event in happening around a topic (here climate strike and conference)  |
| `usr_type` | Type of user who posted the image (here movement and countermovement)                     |

The `event` and `usr_type` values are retained throughout the pipeline and are used when constructing the narrative graphs.

The <description_input_file> must contain at least the following columns:

| Column    | Description                        |
| --------- | ---------------------------------- |
| `Dir`     | Directory containing the image     |
| `ImageID` | Image filename                     |
| `Labels`  | Generated description of the image |




