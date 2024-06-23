import pandas as pd
from transformers import pipeline
import torch
import os 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = 0

translator = pipeline('translation', model='Helsinki-NLP/opus-mt-nl-en', device=device)

def translate_text(text, index, total):
    try:
        translated = translator(text, max_length=3000)
        print(f"Translated {index + 1}/{total} rows.")
        return translated[0]['translation_text']
    except Exception as e:
        print(f"Error translating row {index + 1}: {e}")
        return text

df = pd.read_csv(r"C:\Users\arisa\Downloads\512_or_more_char.csv")

total_rows = len(df)

translated_texts = []

base_path = "C:/Users/arisa/Downloads/data_data_translated_512_or_more"

for idx, row in df.iterrows():
    translated_text = translate_text(row['body'], idx, total_rows)
    translated_texts.append(translated_text)
    df.at[idx, 'body_eng'] = translated_text

    if idx % 2000 == 0 or idx == total_rows - 1:
        save_path = f"{base_path}{idx//2000}.csv"
        df.to_csv(save_path, index=False)
        print(f"Checkpoint saved at row {idx} to {save_path}")

if idx % 2000 != 0:
    final_path = f"{base_path}final.csv"
    df.to_csv(final_path, index=False)
    print("Final version saved at:", final_path)

print("Translation completed and all files saved.")
