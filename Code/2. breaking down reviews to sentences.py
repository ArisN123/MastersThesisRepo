import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def tokenize_reviews_to_sentences(dataframe, column):
    all_sentences = []
    for index, row in dataframe.iterrows():
        review_sentences = sent_tokenize(row[column], language='english')  
        for sentence in review_sentences:
            new_row = row.copy()
            new_row[column] = sentence
            all_sentences.append(new_row)
    return pd.DataFrame(all_sentences)

df = pd.read_csv(r"D:\thesis work\1. data prep\translated_data_clothing_only.csv")

df_sentences = tokenize_reviews_to_sentences(df, 'body_eng')

df_sentences.to_csv(r"D:\thesis work\1. data prep\translated_data_clothing_only_sentences.csv", index=False)
