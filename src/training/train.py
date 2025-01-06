import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model.llm import GPTModel, tokenizeDataset
import tiktoken
import pandas as pd

def train_model(cfg, dataset, device):
  #initalize model
  model = GPTModel(cfg).to(device)

  dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

  #defining loss function and optimizer
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'])

  #training loop
  for epoch in range(cfg['epochs']):
    model.train()
    epoch_loss = 0.
    
    for input_batch, target_batch in dataloader:
      input_batch, target_batch = input_batch.to(device), target_batch.to(device)

      #forward pass
      logits = model(input_batch)
      loss = loss_fn(logits.view(-1, cfg['vocab_size']), target_batch.view(-1))

      #backward pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{cfg['epochs']}, Loss: {epoch_loss / len(dataloader):.4f}")

    # save weights periodically
    if (epoch + 1) % cfg['save_interval'] == 0:
      torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

  print("Training complete!")
  return model

if __name__ == "__main__":
  
  cfg = {
    "vocab_size": 50257,
    "emb_dim": 256,       # Down from 768
    "context_length": 128,  # Down from 256
    "n_layers": 4,        # Down from 12
    "n_heads": 4,         # Down from 12
    "drop_rate": 0.1,
    "batch_size": 4,
    "learning_rate": 5e-4,
    "epochs": 10,          # Down from 10
    "save_interval": 2,
    "qkv_bias": True
  }

  

  
  tweets_path = "data/raw/elon_tweets.csv"
  news_path = "data/raw/finance_news.csv"

  # Read CSVs and handle missing data
  tweets_df = pd.read_csv(tweets_path, on_bad_lines='skip') if os.path.exists(tweets_path) else pd.DataFrame()
  news_df = pd.read_csv(news_path, on_bad_lines='skip') if os.path.exists(news_path) else pd.DataFrame()

  sampled_tweets = tweets_df.sample(frac=0.5, random_state=42) if not tweets_df.empty else pd.DataFrame()
  sampled_news = news_df.sample(frac=0.5, random_state=42) if not news_df.empty else pd.DataFrame()

  # Extract text columns
  tweet_texts = sampled_tweets['text'].tolist() if 'text' in sampled_tweets.columns else []
  news_texts = sampled_news['text'].tolist() if 'text' in sampled_news.columns else []

  # Combine sampled texts
  combined_text = " ".join(tweet_texts + news_texts)

  # Tokenize and prepare dataset
  tokenizer = tiktoken.get_encoding("gpt2")
  dataset = tokenizeDataset(combined_text, tokenizer, cfg['context_length'], stride=128)

  # set device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # train model
  model = train_model(cfg, dataset, device)