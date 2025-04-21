import pandas as pd

def transform_fake_news_dataset(input_file, output_file):
    """
    Transform fake_news_dataset_500.csv according to specified format.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    """
    try:
        # Read the input CSV file
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Successfully read {len(df)} rows")
        
        # Create new transformed dataframe according to required structure
        transformed_df = pd.DataFrame({
            'ID': df['id'],  # Use existing id
            'Modality': ['Text'] * len(df),  # Set all as Text
            'Content': df['content'],  # Use content field directly
            'Label': ['Fake'] * len(df),  # Set all as Fake
            'Source': ['N/A'] * len(df),  # Set all sources as N/A (fictional)
            'Explanation': df['explanation'],  # Use existing explanation
            'Category': df['category'],  # Use existing category
            'Date': df['publication_date']  # Use publication_date
        })
        
        # Write to output CSV
        transformed_df.to_csv(output_file, index=False)
        print(f"Successfully wrote {len(transformed_df)} rows to {output_file}")
        
        # Print some stats
        print("\nCategory distribution in transformed data:")
        print(transformed_df['Category'].value_counts())
        
    except Exception as e:
        print(f"Error during transformation: {e}")

if __name__ == "__main__":
    input_file = "fake_news_dataset_500.csv"
    output_file = "structured_fake_news.csv"
    transform_fake_news_dataset(input_file, output_file)