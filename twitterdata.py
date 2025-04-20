import requests
import csv
from datetime import datetime
import time
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TWITTER_API_KEY")

BASE_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
HEADERS = {"X-API-Key": api_key}

def get_tweets_from_users(usernames, start_date, end_date, query_type="Latest", max_pages_per_user=3):
    all_tweets = []
    
    # Process each username separately to ensure we get tweets from everyone
    for username in usernames:
        print(f"Retrieving tweets from @{username}...")
        
        # Create query for this specific user with date range
        date_filter = f" since:{start_date} until:{end_date}"
        query = f"from:{username}" + date_filter
        
        user_tweets = []
        cursor = ""
        
        for page in range(max_pages_per_user):
            params = {"query": query, "queryType": query_type, "cursor": cursor}
            
            # Add a slight delay to avoid rate limiting
            time.sleep(0.5)
            
            response = requests.get(BASE_URL, headers=HEADERS, params=params)
            if response.status_code != 200:
                print(f"Error {response.status_code} for {username}: {response.text}")
                break
                
            data = response.json()
            tweets = data.get("tweets", [])
            user_tweets.extend(tweets)
            
            print(f"  Page {page+1}: Retrieved {len(tweets)} tweets")
            
            if not data.get("has_next_page"):
                break
                
            cursor = data.get("next_cursor", "")
        
        print(f"Total tweets from @{username}: {len(user_tweets)}\n")
        all_tweets.extend(user_tweets)
    
    return all_tweets

def save_tweets_to_csv(tweets, filename=None):
    if filename is None:
        # Create a filename with timestamp if not provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tweets_{timestamp}.csv"
    
    # Define the fields we want to extract from each tweet
    fieldnames = ['tweet_id', 'text', 'created_at', 'username', 'name', 'retweet_count', 'like_count', 'reply_count']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for tweet in tweets:
            # Extract the desired fields from each tweet
            writer.writerow({
                'tweet_id': tweet.get('id'),
                'text': tweet.get('text', '').replace('\n', ' '),  # Replace newlines to avoid breaking CSV format
                'created_at': tweet.get('created_at', ''),
                'username': tweet.get('author', {}).get('userName', ''),
                'name': tweet.get('author', {}).get('name', ''),
                'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
                'reply_count': tweet.get('public_metrics', {}).get('reply_count', 0)
            })
    
    print(f"Tweets saved to {filename}")
    return filename

def summarize_tweets_by_user(tweets):
    # Count tweets by username
    user_counts = {}
    for tweet in tweets:
        username = tweet.get('author', {}).get('userName', '')
        if username in user_counts:
            user_counts[username] += 1
        else:
            user_counts[username] = 1
    
    # Print summary
    print("\nTweet count by user:")
    for username, count in user_counts.items():
        print(f"@{username}: {count} tweets")

if __name__ == "__main__":
    # List of specific Twitter usernames
    usernames = ["maggieNYT", "guardiannews", "BBCBreaking", "Reuters", "jaketapper"]
    
    # Date range parameters (YYYY-MM-DD format)
    start_date = "2025-03-15"
    end_date = "2025-04-15"
    
    # Get tweets from specific users within date range
    tweets = get_tweets_from_users(usernames, start_date, end_date)
    
    # Print summary information
    print(f"\nRetrieved a total of {len(tweets)} tweets from {len(usernames)} accounts")
    print(f"Date range: {start_date} to {end_date}")
    
    # Summarize tweets by user
    summarize_tweets_by_user(tweets)
    
    # Create a descriptive filename
    filename = f"news_tweets_{start_date}_to_{end_date}.csv"
    save_tweets_to_csv(tweets, filename)
    
    # Print sample of tweets (first 5)
    print("\nSample of retrieved tweets:")
    for i, tweet in enumerate(tweets[:5]):
        print(f"{i+1}. @{tweet['author']['userName']} ({tweet.get('created_at', '')}): {tweet['text'][:100]}...")