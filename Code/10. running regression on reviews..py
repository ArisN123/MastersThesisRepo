import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(r"D:\thesis work\1. data prep\translated_data_clothing_with_topics_wordcount_sentiment.csv")

df = df.dropna()

boolean_columns = ['Quality','Customer Service','Shipping','Size','Style','Price']
df[boolean_columns] = df[boolean_columns].astype(int)

categorical_features = ['purchase_season', 'the_to_type']
encoder = OneHotEncoder(drop='first') 

X_numeric = df[['Word_Count', 'months_between'] + boolean_columns]  
X_categorical = df[categorical_features] 

encoder.fit(X_categorical)
encoded_categorical = encoder.transform(X_categorical).toarray()
encoded_feature_names = encoder.get_feature_names_out(categorical_features)

X_combined = pd.concat([X_numeric, pd.DataFrame(encoded_categorical, columns=encoded_feature_names)], axis=1)

X_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
X_combined.dropna(inplace=True)

X_combined = sm.add_constant(X_combined)

y = df.loc[X_combined.index, 'Sentiment_Score']  

model = sm.OLS(y, X_combined).fit()

print(model.summary())

with open(r"D:\thesis work\extended_regression_summary_v2.txt", "w") as file:
    file.write(model.summary().as_text())
