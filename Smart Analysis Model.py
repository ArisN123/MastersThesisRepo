import pandas as pd
import torch
import os
import numpy as np
import nltk
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from scipy.special import softmax
from nltk.tokenize import sent_tokenize

# Set environment variables and device configuration
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download necessary NLTK packages
nltk.download('punkt')

# Load models and tokenizers
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_model.eval()

attributes = ['Quality', 'Customer Service', 'Shipping', 'Size', 'Style', 'Price']
topic_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
topic_models = {attr: RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2) for attr in attributes}

# Define helper functions
def translate_text(text, translator, index, total):
    try:
        translated = translator(text, max_length=3000)
        print(f"Translated {index + 1}/{total} rows.")
        return translated[0]['translation_text']
    except Exception as e:
        print(f"Error translating row {index + 1}: {e}")
        return text

def tokenize_reviews_to_sentences(dataframe, column):
    all_sentences = []
    for index, row in dataframe.iterrows():
        review_sentences = sent_tokenize(row[column], language='english')
        for sentence in review_sentences:
            new_row = row.copy()
            new_row[column] = sentence
            all_sentences.append(new_row)
    return pd.DataFrame(all_sentences)

def get_sentiment(text):
    text = preprocess(text)
    if text == "":
        return np.array([0.0, 0.0, 0.0])
    encoded_input = sentiment_tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = sentiment_model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
    return scores

def preprocess(text):
    if not isinstance(text, str):
        return ""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def word_count(text):
    return len(str(text).split())

def load_model(attribute):
    model_path = f'C:/Users/arisa/saved_models/{attribute}/model.pth' #Adjust with your own model path
    model = topic_models[attribute]
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Main workflow to process, translate, analyze, and model the data
df = pd.read_csv(r"D:\thesis work\1. data prep\data_clothing_only.csv") #adjust with your own CSV

# Translation
translator = pipeline('translation', model='Helsinki-NLP/opus-mt-nl-en', device=device.index if isinstance(device, torch.device) else 0)
total_rows = len(df)
for idx, row in df.iterrows():
    translated_text = translate_text(row['body'], translator, idx, total_rows)
    df.at[idx, 'body_eng'] = translated_text
    if idx % 2000 == 0 or idx == total_rows - 1:
        save_path = f"D:\\thesis work\\1. data prep\\translated_data_clothing_only_{idx//2000}.csv"
        df.to_csv(save_path, index=False)
        print(f"Checkpoint saved at row {idx} to {save_path}")

# Sentence Tokenization
df_sentences = tokenize_reviews_to_sentences(df, 'body_eng')
df_sentences.to_csv(r"D:\thesis work\1. data prep\translated_data_clothing_only_sentences.csv", index=False) #adjust with your own CSV

# Sentiment Analysis
df_sentences['Sentiment_Scores'] = df_sentences['body_eng'].apply(get_sentiment)
df_sentences['Sentiment'] = np.argmax(df_sentences['Sentiment_Scores'].tolist(), axis=1)
df_sentences['Sentiment_Label'] = df_sentences['Sentiment'].map({0: 'negative', 1: 'neutral', 2: 'positive'})
df_sentences.to_csv(r"D:\thesis work\1. data prep\translated_data_clothing_with_topics_wordcount_sentiment.csv", index=False) #adjust with your own CSV

# Word Count
df_sentences['Word_Count'] = df_sentences['body_eng'].apply(word_count)
df_sentences.to_csv(r"D:\thesis work\1. data prep\translated_data_clothing_with_topics_wordcount.csv", index=False) #adjust with your own CSV

# Apply Models
topic_models = {attr: load_model(attr) for attr in attributes}
for attr in attributes:
    df_sentences[attr] = None

for index, row in df_sentences.iterrows():
    predictions = prepare_data_and_predict(row['body_eng'], topic_tokenizer, topic_models)
    for attr in attributes:
        df_sentences.at[index, attr] = predictions[attr]

df_sentences.to_csv(r'D:\thesis work\1. data prep\translated_data_clothing_with_topics.csv', index=False) #adjust with your own CSV

# Regression Analysis
df_final = pd.read_csv(r"D:\thesis work\1. data prep\translated_data_clothing_with_topics_wordcount_sentiment.csv")  #adjust with your own CSV
df_final = df_final.dropna()
boolean_columns = ['Quality', 'Customer Service', 'Shipping', 'Size', 'Style', 'Price']
df_final[boolean_columns] = df_final[boolean_columns].astype(int)
X_numeric = df_final[boolean_columns + ['Word_Count']]
X_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
X_numeric.dropna(inplace=True)
X_numeric = sm.add_constant(X_numeric)
y = df_final.loc[X_numeric.index, 'Sentiment_Score']
model = sm.OLS(y, X_numeric).fit()
with open(r"D:\thesis work\extended_regression_summary_v2.txt", "w") as file:  #adjust with your own location
    file.write(model.summary().as_text())

