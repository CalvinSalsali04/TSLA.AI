
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.model.llm import GPTModel
import pandas as pd
import tiktoken
from torch.optim.lr_scheduler import StepLR


#config
cfg = {
  "vocab_size": 50257,
  "emb_dim": 256,      
  "context_length": 128,  
  "n_layers": 4,        
  "n_heads": 4,         
  "drop_rate": 0.1,
  "batch_size": 4,
  "learning_rate": 5e-4,
  "epochs": 10,          
  "save_interval": 2,
  "qkv_bias": True
}

class SentimentDataset(Dataset):
  def __init__(self, data_path, tokenizer, max_length):
    self.data = pd.read_csv(data_path)
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.input_ids = []
    self.labels = []

    #loop through each row in csv and tokenize
    for _, row in self.data.iterrows():
      encoded_text = tokenizer.encode(row['text'])
      sentiment = row['Sentiment']
      label = 0 if sentiment == 'negative' else (1 if sentiment == 'positive' else 2)


      if len(encoded_text) > max_length:
        encoded_text = encoded_text[:max_length] # trunicate
      else:
        encoded_text += [0] * (max_length - len(encoded_text)) # pad

      if torch.rand(1).item() < 0.1:  # 10% chance to mask
        encoded_text[torch.randint(0, len(encoded_text), (1,))] = 0


      self.input_ids.append(torch.tensor(encoded_text))
      self.labels.append(torch.tensor(label))

  def __len__(self):
    return len(self.input_ids) # total numb of samples
  
  def __getitem__(self, idx):
    return self.input_ids[idx], self.labels[idx] # tokenized input and label


def fine_tune_model(cfg, train_dataset, val_dataset, device):
  model = GPTModel(cfg).to(device)

  try:
    checkpoint = torch.load("models/pretraining/model_epoch_10.pth", map_location='cpu')
    keys_to_remove = [k for k in checkpoint.keys() if "mask" in k]
    for k in keys_to_remove:
      del checkpoint[k]
    model.load_state_dict(checkpoint)  # Strict loading by default
    print("Loaded pre-trained weights (with mask keys removed).")
  except FileNotFoundError:
    print("No pre-trained model found. Training from scratch.")

  # Replace Output Layer for 3 Classes (Positive, Negative, Neutral)
  model.out_head = nn.Linear(cfg['emb_dim'], 3).to(device)
  nn.init.xavier_uniform_(model.out_head.weight)

  dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=cfg['batch_size'])
  loss_fn = nn.CrossEntropyLoss() # cross-entropy for classification
  optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=0.02) # AdamW optimizer

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.4)

  for epoch in range(cfg['epochs']):
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for input_batch, target_batch in dataloader:
      input_batch, target_batch = input_batch.to(device), target_batch.to(device)

      logits = model(input_batch)
      logits = logits[:, -1, :]  # shape: (batch_size, 3)
      
      # Match target shape
      loss = loss_fn(logits, target_batch)




      # backpropagation
      optimizer.zero_grad()
      loss.backward() 
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()  

      total_loss += loss.item()
      predictions = torch.argmax(logits, dim=-1)
      correct += (predictions == target_batch).sum().item()
      total += target_batch.numel()


    scheduler.step()
    print(f"Epoch {epoch + 1}/{cfg['epochs']}, Loss: {total_loss:.4f}")

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
      for input_batch, target_batch in val_dataloader:
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        logits = logits[:, -1, :]
        
        val_loss += loss_fn(logits, target_batch).item()

        predictions = torch.argmax(logits, dim=-1)
        val_correct += (predictions == target_batch).sum().item()
        val_total += target_batch.size(0)
      
    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    if (epoch + 1) % cfg['save_interval'] == 0:
      torch.save(model.state_dict(), f"fine_tuned_epoch_{epoch + 1}.pth")

  print("Fine-tuning complete.")
  return model

if __name__ == "__main__":
  tokenizer = tiktoken.get_encoding("gpt2")  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

  val_dataset = SentimentDataset("data/processed/labeled_tesla_news.csv", tokenizer, cfg['context_length'])
  train_dataset = SentimentDataset("data/processed/labeled_elon_tweets.csv", tokenizer, cfg['context_length'])
  
  fine_tune_model(cfg, train_dataset, val_dataset, device)
