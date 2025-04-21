import pandas as pd

# Read both CSV files
df1 = pd.read_csv('structured_fake_news.csv')
df2 = pd.read_csv('merged_headlines_and_tweets_dataset.csv')

# Concatenate the dataframes
merged_df = pd.concat([df1, df2], ignore_index=True)

# Remove any duplicate rows if needed
# merged_df = merged_df.drop_duplicates()

# Save the merged dataset to a new CSV file
merged_df.to_csv('merged_dataset.csv', index=False)

print(f"Successfully merged datasets: {len(df1)} rows from first dataset and {len(df2)} rows from second dataset")
print(f"Total rows in merged dataset: {len(merged_df)}")