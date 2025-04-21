import pandas as pd
import re

def transform_headlines_csv(input_file, output_file):
    """
    Transform merged_headlines_20250415_220601.csv according to specified format.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    """
    try:
        # Read the input CSV file
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Successfully read {len(df)} rows")
        
        # Helper function to determine category based on content
        def determine_category(title, description, existing_category):
            """Determine category based on content if existing is 'Unknown'"""
            if existing_category and existing_category != 'Unknown':
                return existing_category
                
            # Combine title and description for analysis
            content = f"{title} {description}".lower()
            
            # Define keyword patterns for categories
            category_patterns = {
                'Politics': ['president', 'trump', 'biden', 'government', 'election', 'congress', 'senate', 'political', 
                            'vote', 'democrat', 'republican', 'administration', 'policy', 'minister', 'parliament'],
                'Economy': ['economy', 'economic', 'market', 'stock', 'inflation', 'recession', 'interest rate', 'fed', 
                           'banking', 'finance', 'gdp', 'unemployment', 'trade', 'tariff'],
                'Health': ['health', 'covid', 'pandemic', 'virus', 'disease', 'medical', 'hospital', 'vaccine', 
                          'medicine', 'doctor', 'outbreak', 'patient', 'healthcare'],
                'Technology': ['tech', 'technology', 'ai', 'artificial intelligence', 'digital', 'app', 'software', 
                              'internet', 'cyber', 'robot', 'innovation', 'computer', 'smartphone'],
                'Environment': ['climate', 'environment', 'carbon', 'emission', 'green', 'renewable', 'pollution', 
                               'sustainable', 'fossil fuel', 'conservation', 'weather', 'disaster'],
                'International': ['international', 'global', 'foreign', 'world', 'country', 'nation', 'border', 'war', 
                                 'peace', 'diplomacy', 'treaty', 'alliance', 'conflict', 'refugee', 'immigration'],
                'Sports': ['sport', 'game', 'team', 'player', 'championship', 'tournament', 'match', 'olympics', 
                          'football', 'soccer', 'basketball', 'tennis', 'score', 'win', 'league'],
                'Entertainment': ['entertainment', 'movie', 'film', 'actor', 'celebrity', 'music', 'concert', 'tv', 
                                 'television', 'streaming', 'award', 'star', 'performance', 'media'],
                'Science': ['science', 'scientific', 'research', 'discovery', 'study', 'scientist', 'space', 'nasa', 
                           'experiment', 'physics', 'biology', 'chemistry', 'astronomy', 'laboratory']
            }
            
            # Check for category keywords
            for category, keywords in category_patterns.items():
                if any(keyword in content for keyword in keywords):
                    return category
            
            return 'Miscellaneous'  # Default if no category determined
        
        # Create new transformed dataframe according to required structure
        transformed_df = pd.DataFrame({
            'ID': df['ID'],  # Use existing ID
            'Modality': ['Text'] * len(df),  # Set all as Text
            'Content': df.apply(lambda row: f"{row['title']} {row['description']}" 
                               if pd.notna(row['description']) else row['title'], axis=1),
            'Label': df['label'],  # Use existing label
            'Source': df['url'],  # Use url as source
            'Explanation': df['explanation'],  # Use existing explanation
            'Category': df.apply(lambda row: determine_category(row['title'], 
                                                               str(row['description']) if pd.notna(row['description']) else "",
                                                               row['category']), axis=1),
            'Date': df['date_collected']  # Use date_collected
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
    input_file = "merged_headlines_20250415_220601.csv"
    output_file = "structured_headlines.csv"
    transform_headlines_csv(input_file, output_file)