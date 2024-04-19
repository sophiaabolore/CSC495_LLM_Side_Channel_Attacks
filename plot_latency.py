import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
# Load CSV file
df = pd.read_csv('latency_demographics_explicit.csv')
df['tokens'] = df['response'].apply(lambda x: len(x.split()))
# Calculate tokens per second 
df['tokens_per_second'] = df['tokens'] / df['latency']

# Plot 1: Latency by Age
plt.figure(figsize=(10, 6))
sns.boxplot(x='age', y='latency', data=df)
plt.title('Latency by Age')
plt.xlabel('Age')
plt.ylabel('Latency (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Tokens per Second by Gender
plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='tokens_per_second', data=df)
plt.title('Tokens per Second by Gender')
plt.xlabel('Gender')
plt.ylabel('Tokens per Second')
plt.tight_layout()
plt.show()

# Plot 3: Latency Distribution by Race
plt.figure(figsize=(10, 6))
sns.violinplot(x='race', y='latency', data=df)
plt.title('Latency Distribution by Race')
plt.xlabel('Race')
plt.ylabel('Latency (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

scaler = MinMaxScaler()
df['normalized_latency'] = scaler.fit_transform(df[['latency']])

# Plot 1: Normalized Latency by Age
plt.figure(figsize=(10, 6))
sns.boxplot(x='age', y='normalized_latency', data=df)
plt.title('Normalized Latency by Age')
plt.xlabel('Age')
plt.ylabel('Normalized Latency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Normalize tokens_per_second for the entire DataFrame
df['normalized_tokens_per_second'] = scaler.fit_transform(df[['tokens_per_second']])

# Plot 2: Normalized Tokens per Second by Gender
plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='normalized_tokens_per_second', data=df)
plt.title('Normalized Tokens per Second by Gender')
plt.xlabel('Gender')
plt.ylabel('Normalized Tokens per Second')
plt.tight_layout()
plt.show()

# Plot 3: Normalized Latency Distribution by Race
plt.figure(figsize=(10, 6))
sns.violinplot(x='race', y='normalized_latency', data=df)
plt.title('Normalized Latency Distribution by Race')
plt.xlabel('Race')
plt.ylabel('Normalized Latency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

average_latency_by_race = df.groupby('race')['latency'].mean()
print("Average Latency by Race:")
print(average_latency_by_race)

# Calculate and print average latency for each combination of gender and age
average_latency_by_gender = df.groupby(['gender'])['latency'].mean()
print("\nAverage Latency by Gender")
print(average_latency_by_gender)

average_latency_by_age = df.groupby(['age'])['latency'].mean()
print("\nAverage Latency by Age")
print(average_latency_by_age)


def classify_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Apply sentiment analysis to the response column
df['sentiment'] = df['response'].apply(classify_sentiment)

# Aggregate and print counts of sentiment categories for each demographic group
# For Race
print("Sentiment Counts by Race:")
print(df.groupby(['race', 'sentiment'])['sentiment'].count())

# For Gender
print("\nSentiment Counts by Gender:")
print(df.groupby(['gender', 'sentiment'])['sentiment'].count())

# For Age
print("\nSentiment Counts by Age:")
print(df.groupby(['age', 'sentiment'])['sentiment'].count())
