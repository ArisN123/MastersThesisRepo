import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Define the device
device = torch.device("cuda")

# Function to load a model given an attribute name
def load_model(attribute, num_labels=2):
    model_path = f'C:/Users/arisa/saved_models/{attribute}/model.pth'
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
# Function to prepare data and predict
def prepare_data_and_predict(text, tokenizer, models):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,  # Ensure this matches training configuration
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    predictions = {}
    with torch.no_grad():
        for attribute, model in models.items():
            outputs = model(input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions[attribute] = pred

    return predictions

# Load tokenizer and models
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
attributes = ['Quality', 'Customer Service', 'Shipping', 'Size', 'Style', 'Price']
models = {attr: load_model(attr) for attr in attributes}

# Function to apply models to dataframe
def apply_models_to_dataframe(df):
    for attr in attributes:
        df[attr] = None  # Initialize new columns for each attribute

    for index, row in df.iterrows():
        predictions = prepare_data_and_predict(row['body_eng'], tokenizer, models)
        for attr in attributes:
            df.at[index, attr] = predictions[attr]

    return df

# Load the dataset
df = pd.read_csv(r'D:\thesis work\1. data prep\translated_data_clothing_only_sentences.csv')

# Apply the models to the dataframe
df = apply_models_to_dataframe(df)

# Save the updated dataframe
df.to_csv(r'D:\thesis work\1. data prep\translated_data_clothing_with_topics.csv', index=False)
