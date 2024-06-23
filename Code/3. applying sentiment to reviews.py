import torch  
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def preprocess(text):
    if not isinstance(text, str): 
        return ""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_sentiment(text):
    text = preprocess(text)
    if text == "": 
        return np.array([0.0, 0.0, 0.0])
    encoded_input = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
    return scores

df = pd.read_csv(r"D:\thesis work\1. data prep\translated_data_clothing_with_topics_wordcount.csv")

sentiment_scores = df['body_eng'].apply(get_sentiment)
df['Sentiment'] = np.argmax(sentiment_scores.tolist(), axis=1)
df['Sentiment_Score'] = np.max(sentiment_scores.tolist(), axis=1)

sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
df['Sentiment'] = df['Sentiment'].map(sentiment_labels)

df.to_csv(r"D:\thesis work\1. data prep\translated_data_clothing_with_topics_wordcount_sentiment.csv", index=False)
print("Updated CSV with sentiment analysis saved successfully.")
