import pandas as pd

# Load the dataset and print the column names
df = pd.read_csv(r"C:\Users\arisa\Downloads\sizing_training.csv", encoding='iso-8859-1')
print(df.columns)  # This will show you all column names
