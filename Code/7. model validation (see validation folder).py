import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

class TextDataDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

def load_model(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

model_path = r"D:\thesis work\DEEPLEARNINGmodel"
model, tokenizer = load_model(model_path)
model.eval()
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

file_paths = [
    r"D:\thesis work\validation\product_1_reviews_modified.csv",
    r"D:\thesis work\validation\Product_2_reviews_modified.csv",
    r"D:\thesis work\validation\product_3_reviews_modified.csv"
]
output_files = [
    r"D:\thesis work\validation\predictions1.csv",
    r"D:\thesis work\validation\predictions2.csv",
    r"D:\thesis work\validation\predictions3.csv"
]

for file_path, output_file in zip(file_paths, output_files):
    df = pd.read_csv(file_path)
    
    dataset = TextDataDataset(texts=df['Product'].tolist(), tokenizer=tokenizer, max_len=128)
    loader = DataLoader(dataset, batch_size=10)

    results = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            attention_mask = batch['attention_mask'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs.logits).cpu().numpy() > 0.45
            results.extend(preds)

    predictions_df = pd.DataFrame(results, columns=['Quality', 'Sizing', 'DesignStyle', 'ProductPhotography', 'PricePoint', 'ShippingOptions', 'CustomerService', 'SustainabilityEthical'])

    result_df = pd.concat([df, predictions_df], axis=1)

    result_df.to_csv(output_file, index=False)
    print(f"Predictions saved successfully to {output_file}.")
