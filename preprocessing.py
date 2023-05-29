# Importing libraries

import re
import os
import spacy
import logging
import argparse
import pandas as pd
from spacy.lang.en import stop_words

# Configurar o nível de registro
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

from tqdm.notebook import tqdm
tqdm.pandas()

logging.info('Loading the english file!!')
os.system('python -m spacy download en_core_web_md')
nlp = spacy.load("en_core_web_md")

def remove_puctuation(doc):
  doc = nlp(doc)
  final_text = [token.text for token in doc if not token.is_punct]
  final_text = ' '.join(final_text)
  return final_text

def remove_stop_words(doc):
  doc = nlp(doc)
  final_text = [token.text for token in doc if not token.is_stop]
  final_text = ' '.join(final_text)
  return final_text

def lemmatization(doc):
  doc = nlp(doc)
  lemmatized_sentence = [token.lemma_ for token in doc]
  lemmatized_sentence = ' '.join(lemmatized_sentence)
  return lemmatized_sentence

def remove_url(text: str): 
  url_pattern  = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
  return url_pattern.sub(r'', text)

def remove_numbers(doc):
  doc = nlp(doc) 
  final_text = [token.text for token in doc if token.is_alpha]
  final_text = ' '.join(final_text)
  return final_text
        
def main(file_path: str, output_file: str):

    train = pd.read_csv(file_path)

    train['review'] = train['review'].progress_apply(lambda x: x.lower())
    #train['review'] = train['review'].apply(lambda x: re.sub(r'[^\w\s]', '', x)) 
    train['review'] = train['review'].progress_apply(lambda x: remove_puctuation(x))
    train['review'] = train['review'].progress_apply(lambda x: lemmatization(x))
    train['review'] = train['review'].progress_apply(lambda x: remove_url(x))
    train['review'] = train['review'].progress_apply(lambda x: remove_stop_words(x))
    train['review'] = train['review'].progress_apply(lambda x: remove_numbers(x))
    
    train.to_csv(output_file)


if __name__ == "__main__":

    # Creating ArgumentParser object
    parser = argparse.ArgumentParser(description='Text preprocessing.')

    # Defining arguments
    parser.add_argument('--file_path', type=str, help='The complete file path of the original .csv file containing the sentences.')
    parser.add_argument('--output_file', type=str, help='The complete output path where the preprocessed texts should be saved (including the extension).')

    args = parser.parse_args()

    file_path = args.file_path
    output_file = args.output_file
    
    main(file_path, output_file)