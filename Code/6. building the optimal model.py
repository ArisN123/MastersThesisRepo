import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW

def calculate_accuracy(true_labels, predictions):
    return (true_labels == predictions).mean()

def train_model(train_csv_file, test_csv_file, attribute, params):
    df_train = pd.read_csv(train_csv_file, encoding='iso-8859-1')
    df_test = pd.read_csv(test_csv_file, encoding='iso-8859-1')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    class ReviewDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
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
                'labels': torch.tensor(label, dtype=torch.long)
            }

    train_dataset = ReviewDataset(df_train['review'].tolist(), df_train[attribute].tolist(), tokenizer)
    test_dataset = ReviewDataset(df_test['Sentence'].tolist(), df_test[attribute].tolist(), tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=params['lr'])

    for epoch in range(params['epochs']):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch['labels'].cpu().numpy())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    accuracy = calculate_accuracy(test_labels, test_preds)
    print(f"Accuracy for {attribute}: {accuracy:.4f}")

    model_dir = os.path.join("saved_models", attribute)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model for {attribute} saved to {model_path}")

optimal_params = {
    'Quality': {'batch_size': 8, 'lr': 9.30766636389501e-05, 'epochs': 5},
    'Customer Service': {'batch_size': 16, 'lr': 4.325886142619659e-05, 'epochs': 1},
    'Shipping': {'batch_size': 8, 'lr': 9.972857860144901e-05, 'epochs': 2},
    'Size': {'batch_size': 32, 'lr': 1.3018773393879134e-05, 'epochs': 3},
    'Style': {'batch_size': 16, 'lr': 7.300658729907797e-05, 'epochs': 7},
    'Price': {'batch_size': 8, 'lr': 2.207985471946546e-05, 'epochs': 5}
}

test_csv = r"D:\thesis work\model 3 (mini models)\all_testing.csv"
csv_files = [
    (r"D:\thesis work\model 3 (mini models)\quality_training.csv", 'Quality'),
    (r"D:\thesis work\model 3 (mini models)\CS_training.csv", 'Customer Service'),
    (r"D:\thesis work\model 3 (mini models)\shipping_training.csv", 'Shipping'),
    (r"D:\thesis work\model 3 (mini models)\sizing_training.csv", 'Size'),
    (r"D:\thesis work\model 3 (mini models)\style_training.csv", 'Style'),
    (r"D:\thesis work\model 3 (mini models)\price_training.csv", 'Price')
]

for csv_file, attribute in csv_files:
    train_model(csv_file, test_csv, attribute, optimal_params[attribute])
