from newsapi import NewsApiClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
newsapi_key = os.getenv('NEWSAPI_KEY')
newsapi = NewsApiClient(api_key=newsapi_key)

def get_tesla_news(query="Tesla", max_article=100):
  articles = newsapi.get_everything(
    q=query,
    language='en',
    sort_by='relevancy' # need relevant articles
  )

  news_data = [{
    "date": article['publishedAt'],
    "content": article['title'] + '' + (article['description'] or "")
  } for article in articles['articles'][:max_article]]

  dataframe = pd.DataFrame(news_data)

  dataframe.to_csv("data/raw/tesla_news_recent.csv", index=False)
  print(f"Collected {len(dataframe)} Tesla news articles.")

get_tesla_news()