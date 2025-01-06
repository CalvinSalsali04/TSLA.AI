from twilio.rest import Client
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from datetime import datetime
import pandas as pd
import torch
from src.model.llm import GPTModel
import tiktoken
import schedule
import time
import torch
import torch.nn as nn
from dotenv import load_dotenv



# Twilio Config
load_dotenv()

account_sid = os.getenv('TWILIO_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
from_phone = os.getenv('FROM_PHONE')
to_phone = os.getenv('TO_PHONE')

client = Client(account_sid, auth_token)

# Model Config
model_path = "models/finetuning/fine_tuned_epoch_10.pth"
cfg = {
  "vocab_size": 50257,
  "emb_dim": 256,
  "context_length": 128,
  "n_layers": 4,
  "n_heads": 4,
  "drop_rate": 0.1,
  "qkv_bias": True
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Model Loading
import torch
import torch.nn as nn

def load_model(cfg, checkpoint_path, device):
  print("Initializing GPTModel...")
  model = GPTModel(cfg).to(device)
  print("Model instantiated.")

  # 1) Replace the output head with a 3-class layer
  model.out_head = nn.Linear(cfg["emb_dim"], 3).to(device)
  nn.init.xavier_uniform_(model.out_head.weight)
  print("Replaced output head with 3-class layer.")

  try:
    print("Loading model checkpoint...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    print("loaded")

    # 2) Load the checkpoint. The shapes should now match your [3, 256] finetuned head.
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys in checkpoint: {missing_keys}")
    print(f"Extra keys in checkpoint: {unexpected_keys}")

    model.eval()
    print("Model weights loaded successfully.")
  except Exception as e:
    print(f"Error loading model: {e}")
    raise e

  return model


model = load_model(cfg, model_path, device)


# Data Scraping Function
def scrape_latest_data():
  print("Fetching latest data...")
  try:
    os.system("python3 src/data_collection/scrape_tweets.py")
  except Exception as e:
    print(f"Tweepy API limit hit. Skipping tweet scraping... Error: {e}")

  try:
    os.system("python3 src/data_collection/scrape_news.py")
    os.system("python3 src/data_collection/fetch_stock.py")
    print("Data scraping complete.")
  except Exception as e:
    print(f"Error during scraping: {e}")
    message = client.messages.create(
      from_=from_phone,
      body=f"Error during data scraping: {str(e)}",
      to=to_phone
    )
    print(f"Error SMS sent: {message.sid}")


# Batch Prediction Function
def predict_batch(texts):
  print(f"Running batch prediction for {len(texts)} samples...")

  # Encode and pad inputs
  input_ids = [tokenizer.encode(text) for text in texts]
  max_len = max(len(ids) for ids in input_ids)
  padded = [ids + [0] * (max_len - len(ids)) for ids in input_ids]
  input_tensor = torch.tensor(padded).to(device)

  # Model inference
  with torch.no_grad():
    logits = model(input_tensor)
    predictions = torch.argmax(logits[:, -1, :], dim=-1).tolist()
  
  print("Batch prediction complete.")
  return predictions


# Predict and Send Results
def predict_and_send():
  print("Running prediction...")

  try:
    # Load data
    tweets_path = "data/raw/elon_tweets_recent.csv"
    news_path = "data/raw/tesla_news_recent.csv"
    
    tweets = pd.read_csv(tweets_path) if os.path.exists(tweets_path) else pd.DataFrame()
    news = pd.read_csv(news_path) if os.path.exists(news_path) else pd.DataFrame()

    if tweets.empty and news.empty:
      raise ValueError("No data available for prediction!")

    all_sentiments = []

    # Predict tweets
    if not tweets.empty:
      print(f"Predicting {len(tweets)} tweets...")
      tweet_sentiments = predict_batch(tweets["content"].tolist())
      all_sentiments.extend(tweet_sentiments)

    # Predict news
    if not news.empty:
      print(f"Predicting {len(news)} news articles...")
      news_sentiments = predict_batch(news["content"].tolist())
      all_sentiments.extend(news_sentiments)

    if not all_sentiments:
      raise ValueError("No valid data points for prediction.")

    # Average sentiment to get decision
    average_sentiment = sum(all_sentiments) / len(all_sentiments)
    decision = ["Sell", "Buy", "Hold"][int(average_sentiment)]

    # Save to CSV
    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "prediction": decision
    }
    result_df = pd.DataFrame([result])
    prediction_path = "data/predictions/tsla_predictions.csv"
    result_df.to_csv(prediction_path, mode='a', index=False, header=not os.path.exists(prediction_path))

    # Send SMS
    message_body = f"Tesla Prediction for {result['date']}: {decision} 🚀"
    message = client.messages.create(
      from_=from_phone,
      body=message_body,
      to=to_phone
    )
    print(f"SMS sent! Prediction: {decision}, SID: {message.sid}")

  except Exception as e:
    print(f"Prediction failed: {e}")
    message = client.messages.create(
      from_=from_phone,
      body=f"Error during prediction: {str(e)}",
      to=to_phone
    )
    print(f"Error SMS sent: {message.sid}")


# Scheduler for Automation
def start_scheduler():
  print("Scheduler started... Waiting for next job.", flush=True)
  schedule.every().day.at("15:05").do(scrape_latest_data)
  schedule.every().day.at("15:10").do(predict_and_send)

  while True:
    print("Waiting for next job...", flush=True)
    schedule.run_pending()
    time.sleep(10)


if __name__ == "__main__":
  print("Script is running.")
  start_scheduler()
