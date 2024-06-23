import pandas as pd

df = pd.read_csv(r"D:\thesis work\1. data prep\translated_data_clothing_with_topics.csv")

def word_count(text):
    return len(str(text).split())

df['Word_Count'] = df['body_eng'].apply(word_count)

df.to_csv(r"D:\thesis work\1. data prep\translated_data_clothing_with_topics_wordcount.csv", index=False)
print("Updated CSV with word count saved successfully.")
