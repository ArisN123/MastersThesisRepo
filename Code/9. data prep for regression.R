
translated_data_clothing_with_topics_wordcount_sentiment2<-translated_data_clothing_with_topics_wordcount_sentiment
translated_data_clothing_with_topics_wordcount_sentiment2$the_to_type <- ifelse(translated_data_clothing_with_topics_wordcount_sentiment$the_to_type == "online", 1, 0)

translated_data_clothing_with_topics_wordcount_sentiment2$tdt_type_detail <- ifelse(translated_data_clothing_with_topics_wordcount_sentiment$tdt_type_detail == "return", 1, 0)

translated_data_clothing_with_topics_wordcount_sentiment2$qty <- abs(translated_data_clothing_with_topics_wordcount_sentiment$qty)
translated_data_clothing_with_topics_wordcount_sentiment2$turnover <- abs(translated_data_clothing_with_topics_wordcount_sentiment$turnover)

translated_data_clothing_with_topics_wordcount_sentiment2$language <- ifelse(translated_data_clothing_with_topics_wordcount_sentiment$language == "en", 1, 0)

translated_data_clothing_with_topics_wordcount_sentiment2$Spring <- ifelse(translated_data_clothing_with_topics_wordcount_sentiment$purchase_season == "Spring", 1, 0)
translated_data_clothing_with_topics_wordcount_sentiment2$Summer <- ifelse(translated_data_clothing_with_topics_wordcount_sentiment$purchase_season == "Summer", 1, 0)
translated_data_clothing_with_topics_wordcount_sentiment2$Fall <- ifelse(translated_data_clothing_with_topics_wordcount_sentiment$purchase_season == "Fall", 1, 0)
translated_data_clothing_with_topics_wordcount_sentiment2$Winter <- ifelse(translated_data_clothing_with_topics_wordcount_sentiment$purchase_season == "Winter", 1, 0)

head(translated_data_clothing_with_topics_wordcount_sentiment2)

translated_data_clothing_with_topics_wordcount_sentiment2$turnover <- scale(translated_data_clothing_with_topics_wordcount_sentiment$turnover)
translated_data_clothing_with_topics_wordcount_sentiment2$qty <- scale(translated_data_clothing_with_topics_wordcount_sentiment$qty)
translated_data_clothing_with_topics_wordcount_sentiment2$months_between <- scale(translated_data_clothing_with_topics_wordcount_sentiment$months_between)
translated_data_clothing_with_topics_wordcount_sentiment2$Sentiment_Score <- scale(translated_data_clothing_with_topics_wordcount_sentiment$Sentiment_Score)

head(translated_data_clothing_with_topics_wordcount_sentiment2)



library(dplyr)

data <- translated_data_clothing_with_topics_wordcount_sentiment2

model <- lm(Sentiment_Score ~ Quality + Customer.Service + Shipping + Size + Style + Price + language + Spring + Fall + Winter + months_between + turnover + qty + Word_Count + tdt_type_detail + the_to_type, data = data)

summary(model)



library(ggplot2)
library(dplyr)
library(lubridate)

monthly_data <- translated_data_clothing_with_topics_wordcount_sentiment2 %>%
  mutate(the_date_transaction = as.Date(the_date_transaction, "%Y-%m-%d")) %>%
  mutate(year = year(the_date_transaction), month = month(the_date_transaction)) %>%
  group_by(year, month) %>%
  summarise(median_sentiment_score = median(Sentiment_Score, na.rm = TRUE)) %>%
  ungroup() %>%
  arrange(year, month) %>%
  mutate(date = make_date(year, month, 1))  
# Plotting
ggplot(monthly_data, aes(x = date, y = median_sentiment_score)) +
  geom_line(color = "blue", size = 1) +
  labs(title = "Median Sentiment Score Over Time (Monthly Basis)",
       x = "Date",
       y = "Median Sentiment Score") +
  theme_minimal()



library(ggplot2)
library(dplyr)
library(lubridate)

quality_data <- translated_data_clothing_with_topics_wordcount_sentiment2 %>%
  filter(Quality == 1) %>%
  mutate(the_date_transaction = as.Date(the_date_transaction, "%Y-%m-%d")) %>%
  mutate(year = year(the_date_transaction), month = month(the_date_transaction)) %>%
  group_by(year, month) %>%
  summarise(average_sentiment_score = mean(Sentiment_Score, na.rm = TRUE)) %>%
  ungroup() %>%
  arrange(year, month) %>%
  mutate(date = make_date(year, month, 1)) 
ggplot(quality_data, aes(x = date, y = average_sentiment_score)) +
  geom_line(color = "blue", size = 1) +
  labs(title = "Average Sentiment Score Over Time (Quality = 1)",
       x = "Date",
       y = "Average Sentiment Score") +
  theme_minimal()


model2 <- lm(Sentiment_Score ~ Quality + Customer.Service + Shipping + Size + Style + Price + language + Spring + Fall + Winter + months_between + turnover + qty + Word_Count + tdt_type_detail + the_to_type, 
            data = data, 
            subset = (language == 0))
summary(model2)
