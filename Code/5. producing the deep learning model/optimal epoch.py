import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import optuna
import os

def train_model(train_csv_file, test_csv_file, attribute, batch_size, learning_rate):
    # Load the training dataset
    df_train = pd.read_csv(train_csv_file, encoding='iso-8859-1')
    # Load the test dataset
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

    def objective(trial):
        n_epochs = trial.suggest_int('n_epochs', 1, 10)  # Suggesting epoch range from 1 to 10

        train_dataset = ReviewDataset(df_train['review'].tolist(), df_train[attribute].tolist(), tokenizer)
        test_dataset = ReviewDataset(df_test['Sentence'].tolist(), df_test[attribute].tolist(), tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" )
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
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
        test_preds, test_labels_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels_list.extend(batch['labels'].cpu().numpy())

        accuracy = (np.array(test_preds) == np.array(test_labels_list)).mean()
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=3)

    print(f"Best epoch number for {attribute}: {study.best_trial.params['n_epochs']}")
    print(f"Best accuracy for {attribute} with {study.best_trial.params['n_epochs']} epochs: {study.best_trial.value}")

# Configuration for each attribute with their optimal batch size and learning rate
config = [
    #(r"D:\thesis work\model 3 (mini models)\quality_training.csv", 'Quality', 8, 6.567742162956452e-05), ##Best trial for Quality: 0.851063829787234 Best parameters for Quality: batch_size': 8, 'lr': 9.30766636389501e-05, 'n_epochs': 5}
    (r"D:\thesis work\model 3 (mini models)\CS_training.csv", 'Customer Service', 16, 4.325886142619659e-05),
    (r"D:\thesis work\model 3 (mini models)\shipping_training.csv", 'Shipping', 8, 9.972857860144901e-05),
    (r"D:\thesis work\model 3 (mini models)\sizing_training.csv", 'Size', 32, 1.3018773393879134e-05),
    (r"D:\thesis work\model 3 (mini models)\style_training.csv", 'Style', 16, 7.300658729907797e-05),
    (r"D:\thesis work\model 3 (mini models)\price_training.csv", 'Price', 8, 2.207985471946546e-05),
]

test_csv = r"D:\thesis work\model 3 (mini models)\all_testing.csv"
for file_path, attribute, batch, lr in config:
    train_model(file_path, test_csv, attribute, batch, lr)
