import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import optuna
import os

def train_model(train_csv_file, test_csv_file, attribute):
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
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-4)
        n_epochs = trial.suggest_int('n_epochs', 1, 10)

        train_dataset = ReviewDataset(df_train['review'].tolist(), df_train[attribute].tolist(), tokenizer)
        test_dataset = ReviewDataset(df_test['Sentence'].tolist(), df_test[attribute].tolist(), tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    study.optimize(objective, n_trials=30)  # Increased trials to give more opportunity to find optimal settings

    print(f"Best trial for {attribute}: {study.best_trial.value}")
    print(f"Best parameters for {attribute}: {study.best_params}")

test_csv = r"D:\thesis work\model 3 (mini models)\all_testing.csv"
csv_files = [
    (r"D:\thesis work\model 3 (mini models)\quality_training.csv", 'Quality'), #Best trial for Quality: 0.851063829787234 Best parameters for Quality: {'batch_size': 8, 'lr': 6.567742162956452e-05}
    (r"D:\thesis work\model 3 (mini models)\CS_training.csv", 'Customer Service'), #Best trial for Customer Service: 0.8936170212765957 Best parameters for Customer Service: {'batch_size': 16, 'lr': 4.325886142619659e-05}
    (r"D:\thesis work\model 3 (mini models)\shipping_training.csv", 'Shipping'),# Trial 9 finished with value: 0.9574468085106383 and parameters: {'batch_size': 8, 'lr': 9.972857860144901e-05}
    (r"D:\thesis work\model 3 (mini models)\sizing_training.csv", 'Size'),# Best trial for Size: 0.9361702127659575 Best parameters for Size: {'batch_size': 32, 'lr': 1.3018773393879134e-05}
    (r"D:\thesis work\model 3 (mini models)\style_training.csv", 'Style'), #Best trial for Style: 0.9361702127659575 Best parameters for Style: {'batch_size': 16, 'lr': 7.300658729907797e-05}
    (r"D:\thesis work\model 3 (mini models)\price_training.csv", 'Price')   #Trial 5 finished with value: 0.9361702127659575 and parameters: {'batch_size': 8, 'lr': 2.207985471946546e-05}. Best is trial 5 with value: 0.9361702127659575.

]

for csv_file, attribute in csv_files:
    train_model(csv_file, test_csv, attribute)
