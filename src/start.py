import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.texts_processing import TextsTokenizer
from src.data_types import Parameters


with open(os.path.join(os.getcwd(), "data", "config.json"), "r") as jf:
    config_dict = json.load(jf)


# vectorizer = SentenceTransformer(os.path.join("models", "all_sys.transformers"))
vectorizer = SentenceTransformer('distiluse-base-multilingual-cased-v1')

parameters = Parameters.parse_obj(config_dict)

stopwords = []
if parameters.stopwords_files:
    for filename in parameters.stopwords_files:
        root = os.path.join(os.getcwd(), "data", filename)
        stopwords_df = pd.read_csv(root, sep="\t")
        stopwords += list(stopwords_df["stopwords"])


mystem_path = os.path.join(os.getcwd(), "models", "mystem")
# tokenizer = TextsTokenizer(mystem_path)
tokenizer = TextsTokenizer()
tokenizer.add_stopwords(stopwords)

