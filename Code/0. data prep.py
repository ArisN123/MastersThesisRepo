import pandas as pd
data = pd.read_parquet("C:/Users/arisa/Downloads/stg_transaction_detail_last_two_years_Erasmus.parquet")
data2 = pd.read_parquet("C:/Users/arisa/Downloads/Decathlon Data/stg_opv_review__review_Erasmus.parquet")
combined_data = pd.merge(data, data2, left_on='sanitized_transaction_id', right_on='sanitized_ticket_number', how='outer')
print(combined_data.head())
