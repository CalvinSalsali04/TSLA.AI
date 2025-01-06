import cohere
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
cohere_api_key = os.getenv('COHERE_API_KEY')
co = cohere.Client(cohere_api_key)

tweets_dataframe = pd.read_csv("../data/raw/elon_tweets.csv")
news_dataframe = pd.read_csv("../data/raw/tesla_news.csv")

class Example:
  def __init__(self, text, label):
    self.text = text
    self.label = label

examples = [
  # Positive Examples
  Example("Tesla stock is soaring after record earnings!", "positive"),
  Example("Elon Musk announces new AI product, sparking excitement.", "positive"),
  Example("Tesla’s market cap hits an all-time high.", "positive"),
  Example("Tesla launches new battery technology next year.", "positive"),
  Example("Musk hints at Dogecoin payments for Tesla products.", "positive"),
  Example("Tesla partners with Panasonic for battery production.", "positive"),
  Example("Tesla announces expansion of its Gigafactory.", "positive"),
  Example("Tesla’s AI Day presentation impresses analysts.", "positive"),
  Example("Elon Musk tweets 'I love humanity!' after successful launch.", "positive"),
  Example("Tesla's solar roof becomes widely adopted.", "positive"),
  Example("Elon Musk tweets 'Tesla will accept Bitcoin again once mining is more sustainable.'", "positive"),
  Example("Elon Musk tweets 'AI will solve most of our problems, including scarcity.'", "positive"),
  Example("Musk tweets 'Building a Gigafactory in Berlin!'", "positive"),
  Example("Elon Musk confirms 'Starlink will IPO when cash flow is predictable.'", "positive"),
  Example("Musk tweets 'Optimism is the best reality.'", "positive"),
  Example("Tesla officially breaks ground on Texas Gigafactory!", "positive"),
  Example("Musk posts 'Tesla Model S Plaid is the fastest production car ever made.'", "positive"),
  Example("Tesla will open up its Supercharger network to all EVs.", "positive"),
  Example("Tesla is leading the charge in renewable energy. Proud moment.", "positive"),


  # Negative Examples
  Example("Tesla is facing delays in Cybertruck production.", "negative"),
  Example("Elon Musk's tweet hints at layoffs at Tesla.", "negative"),
  Example("Reports indicate Tesla stock could dip.", "negative"),
  Example("Tesla accused of misleading investors about Model 3 production.", "negative"),
  Example("Tesla fined for environmental violations.", "negative"),
  Example("Elon Musk tweets 'Tesla stock price is too high imo' causing shares to plummet.", "negative"),
  Example("Musk tweets about ongoing supply chain disruptions.", "negative"),
  Example("Tesla faces backlash for recalling thousands of cars.", "negative"),
  Example("Elon Musk jokes about bankruptcy during tough quarter.", "negative"),
  Example("Elon tweets 'Selling almost all physical possessions, will own no house.'", "negative"),
  Example("Tesla faces lawsuit over false advertising.", "negative"),
  Example("Elon Musk tweets about not trusting the SEC.", "negative"),
  Example("Elon Musk tweets 'Tesla stock price is too high imo' and shares drop immediately.", "negative"),
  Example("Musk tweets 'We are overproducing, layoffs imminent.'", "negative"),
  Example("Elon Musk tweets 'SEC stands for Shortseller Enrichment Commission.'", "negative"),
  Example("Musk posts 'Some people will never be happy no matter what you do.'", "negative"),
  Example("Elon Musk says 'Filing for bankruptcy is not off the table.'", "negative"),
  Example("Musk tweets 'Funding secured' triggering an SEC investigation.", "negative"),
  Example("Tesla’s production line is halted due to supplier issues.", "negative"),
  Example("Elon Musk says 'Don’t buy Tesla stock if volatility is scary.'", "negative"),
  Example("Musk tweets 'A bunch of angry people are coming after me. Oh well.'", "negative"),


  # Neutral Examples
  Example("Elon Musk says Tesla will focus on autonomous vehicles.", "neutral"),
  Example("Tesla launches a new software update.", "neutral"),
  Example("Tesla’s shareholder meeting is next week.", "neutral"),
  Example("Elon Musk teases a new SpaceX launch.", "neutral"),
  Example("Tesla adds a new color option to Model Y.", "neutral"),
  Example("Musk tweets about Mars colonization plans.", "neutral"),
  Example("Tesla robot prototype revealed, but with no timeline for release.", "neutral"),
  Example("Elon Musk tweets a photo of his pet dog.", "neutral"),
  Example("Elon tweets 'Tesla earnings call tomorrow at 5 PM PT.'", "neutral"),
  Example("Tesla announces new interior upgrades for Model 3.", "neutral"),
  Example("Musk posts 'Tesla Model Y now available globally.'", "neutral"),
  Example("Elon tweets 'Mars launch window opens in 2025.'", "neutral"),
  Example("Tesla reveals quarterly results with mixed reviews.", "neutral"),
  Example("Elon tweets 'Tesla FSD Beta now available in Europe.'", "neutral"),
  Example("Tesla will host AI Day this summer.", "neutral"),
  Example("Tesla releases the Cybertruck design preview.", "neutral"),
  Example("Musk tweets 'Another day, another delivery record.'", "neutral"),


  # Controversial Positive (positive tone but potentially sensitive topics)
  Example("Elon tweets 'Free speech is essential to democracy.'", "positive"),
  Example("Elon Musk defends buying Twitter to protect free speech.", "positive"),
  Example("Musk tweets 'I believe innovation thrives in chaos.'", "positive"),
  Example("Elon Musk defends crypto adoption despite market volatility.", "positive"),
  Example("Tesla’s bold move into AI is controversial but innovative.", "positive"),

  # Controversial Negative (controversial, sarcastic, or market-moving tweets)
  Example("Elon Musk tweets 'Let that sink in' after entering Twitter HQ.", "negative"),
  Example("Elon Musk tweets 'Pronouns suck' sparking backlash.", "negative"),
  Example("Tesla faces heat after Musk tweets about cutting 10% of workforce.", "negative"),
  Example("Elon Musk jokes 'Tesla will create a flamethrower', stirring safety concerns.", "negative"),
  Example("Musk tweets 'Why is everyone so serious?' after deleting a controversial post.", "negative"),
  Example("Elon Musk tweets 'I might create my own social media platform.'", "negative"),

  # Controversial (Humorous, Sarcastic, or Edgy):
  Example("Elon Musk tweets 'Tesla tequila is now sold out. Sorry!'", "negative"),
  Example("Musk tweets 'I’m selling all my houses. Will own no home.'", "negative"),
  Example("Elon posts 'I will donate $6B to solve world hunger if the UN shows me how.'", "neutral"),
  Example("Musk tweets 'I challenge Vladimir Putin to single combat. The stakes? Ukraine.'", "negative"),
  Example("Elon tweets 'Tesla is creating a humanoid robot. What could go wrong?'", "negative"),
  Example("Musk jokes 'Tesla will start producing catgirls next year.'", "neutral"),
  Example("Elon Musk tweets 'Pronouns suck' sparking controversy.", "negative"),
  Example("Musk posts 'We need less drama and more work ethic.'", "negative"),
  Example("Elon tweets 'Some mornings I wake up and think about memes.'", "neutral"),
  Example("Elon Musk jokes 'Tesla Roadster will fly with SpaceX thrusters.'", "positive"),
  Example("Musk tweets 'Tesla is raising car prices due to inflation pressures.'", "negative"),
  Example("Musk tweets 'Next product launch is a cyberpunk-inspired whistle.'", "neutral"),
  Example("Elon Musk jokes 'I'm thinking of quitting my job and becoming an influencer.'", "neutral"),

]

def batch_sentiment_analysis(texts, batch_size=96):  #cohere has a batch limit of 96
  results = []
  for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    try:
      response = co.classify(
        model='large', 
        inputs=batch,
        examples=examples
      )
      results.extend([result.prediction for result in response.classifications])
    except Exception as e:
      print(f"Batch processing error: {str(e)}")
      results.extend(["unknown"] * len(batch))
  return results


tweets_dataframe['Sentiment'] = batch_sentiment_analysis(tweets_dataframe['text'].tolist())
news_dataframe['Sentiment'] = batch_sentiment_analysis(news_dataframe['text'].tolist())

tweets_dataframe.to_csv("../data/processed/labeled_elon_tweets.csv", index=False)
news_dataframe.to_csv("../data/processed/labeled_tesla_news.csv", index=False)


print("Labeled Elon Musk tweets and Tesla news articles with sentiment.")