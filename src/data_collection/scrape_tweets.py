import tweepy
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
client = tweepy.Client(bearer_token=bearer_token)


def scrape_elon_tweets(max_tweets=100):
  user_id = '44196397'  
    
  response = client.get_users_tweets(
    id=user_id,
    max_results=min(max_tweets, 100),  
    tweet_fields=['created_at']
  )
  
  
  if response.data:
    tweet_list = [[tweet.created_at, tweet.text] for tweet in response.data]
    df = pd.DataFrame(tweet_list, columns=["Date", "Tweet"])
    df.to_csv("data/raw/elon_tweets_recent.csv", index=False)
    print(f"Scraped {len(df)} tweets from Elon Musk.")
  else:
    print("No tweets found or access is limited.")

scrape_elon_tweets()