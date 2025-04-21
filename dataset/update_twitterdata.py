import pandas as pd
import re

def transform_news_tweets(input_file, output_file):
    """
    Transform news tweets CSV into structured format with specific fields.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    """
    # Read the input CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully read {len(df)} rows from {input_file}")
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return
    
    # List of credible news sources
    credible_sources = [
        'BBCBreaking', 'BBCWorld', 'CNN', 'CNNBreaking', 'nytimes', 
        'WSJ', 'Reuters', 'AP', 'washingtonpost', 'Bloomberg',
        'FT', 'TheEconomist', 'ABC', 'CBSNews', 'NBCNews',
        'NPR', 'PBS', 'guardian', 'AJEnglish', 'CBCNews',
        'maggieNYT', 'jaketapper', 'andersoncooper', 'camanpour', 'maddow'
    ]
    
    # Category keywords dictionary
    category_keywords = {
        'Politics': ['president', 'trump', 'congress', 'senate', 'election', 'vote', 'democracy', 
                    'government', 'administration', 'republican', 'democrat', 'policy', 'bill', 'law'],
        'Economy': ['economy', 'economic', 'market', 'stock', 'inflation', 'recession', 'jobs', 
                   'unemployment', 'interest rate', 'fed', 'federal reserve', 'trade', 'gdp', 
                   'finance', 'banking'],
        'Health': ['covid', 'pandemic', 'health', 'medicine', 'disease', 'hospital', 'doctor', 
                  'patient', 'vaccine', 'treatment', 'medical', 'virus', 'infection', 'outbreak'],
        'Technology': ['tech', 'technology', 'ai', 'artificial intelligence', 'digital', 'cyber', 
                      'internet', 'app', 'software', 'hardware', 'robot', 'automation', 'innovation'],
        'Environment': ['climate', 'environment', 'pollution', 'carbon', 'emissions', 'green', 
                       'sustainability', 'renewable', 'fossil fuel', 'weather', 'storm', 'hurricane', 
                       'biodiversity'],
        'International': ['international', 'global', 'world', 'foreign', 'embassy', 'diplomat', 
                         'treaty', 'country', 'nation', 'border', 'immigration', 'refugee', 'war', 
                         'peace', 'conflict'],
        'Sports': ['sport', 'game', 'team', 'player', 'championship', 'tournament', 'league', 
                  'match', 'score', 'win', 'lose', 'olympic', 'football', 'soccer', 'basketball', 
                  'baseball'],
        'Entertainment': ['entertainment', 'movie', 'film', 'actor', 'actress', 'celebrity', 
                         'music', 'concert', 'album', 'award', 'star', 'TV', 'show', 'series'],
        'Science': ['science', 'research', 'study', 'discovery', 'scientist', 'lab', 'experiment', 
                   'space', 'nasa', 'mission', 'planet', 'star', 'galaxy', 'quantum', 'physics']
    }
    
    def determine_category(text):
        """Determine category based on tweet content"""
        text_lower = text.lower()
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return category
        
        return 'Miscellaneous'  # Default category
    
    def craft_explanation(username, name):
        """Craft explanation based on username and name"""
        if username in credible_sources:
            # For known news organizations
            if username in ['BBCBreaking', 'BBCWorld', 'CNN', 'CNNBreaking', 'Reuters', 'AP']:
                return f"Tweet by {name}, a credible news organization known for factual reporting."
            
            # For newspapers
            if username in ['nytimes', 'WSJ', 'washingtonpost', 'guardian']:
                return f"Tweet by {name}, a reputable newspaper with editorial standards."
            
            # For financial news
            if username in ['Bloomberg', 'FT', 'TheEconomist']:
                return f"Tweet by {name}, a trusted source for financial and business news."
            
            # For broadcast networks
            if username in ['ABC', 'CBSNews', 'NBCNews', 'NPR', 'PBS', 'AJEnglish', 'CBCNews']:
                return f"Tweet by {name}, an established broadcast news network."
            
            # For journalists
            if username in ['maggieNYT', 'jaketapper', 'andersoncooper', 'camanpour', 'maddow']:
                return f"Tweet by {name}, a professional journalist with a reputation for factual reporting."
        
        # Default for other sources
        return f"Tweet by {name} (@{username}), verification based on account history and engagement metrics."
    
    # Create a new DataFrame with the transformed structure
    transformed_df = pd.DataFrame({
        'ID': range(200, 200 + len(df)),
        'Modality': ['Text'] * len(df),
        'Content': df['text'],
        'Label': df['username'].apply(lambda x: 'Real' if x in credible_sources else 'Requires Verification'),
        'Source': ['N/A'] * len(df),  # No direct URLs in the CSV
        'Explanation': df.apply(lambda row: craft_explanation(row['username'], row['name']), axis=1),
        'Category': df['text'].apply(determine_category),
        'Date': df['created_at']  # Using original created_at timestamp
    })
    
    # Write to output CSV
    try:
        transformed_df.to_csv(output_file, index=False)
        print(f"Successfully wrote {len(transformed_df)} rows to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")
        return
    
    print("Transformation completed successfully")

# Example usage
if __name__ == "__main__":
    input_file = "news_tweets_2025-03-15_to_2025-04-15.csv"
    output_file = "structured_news_tweets.csv"
    transform_news_tweets(input_file, output_file)