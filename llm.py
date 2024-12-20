import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class tokenizeDataset(Dataset):
  def __init__(self, txt, tokenizer, max_length, stride):
    self.input_ids = []
    self.target_ids = []

    # tokenize our input data
    token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

    # sliding window to chunk txt
    for i in range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i:i+max_length]
      target_chunk = token_ids[i+1:i+max_length+1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]
  
def createDataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,num_workers=0):

  #initialize tokenizer and create the dataset
  tokenizer = tiktoken.get_encoding("gpt2")
  dataset = tokenizeDataset(txt, tokenizer, max_length, stride)

  #create dataloader
  dataloader = DataLoader(
    dataset,
    batch_size = batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
  )

  return dataloader


# nn.module is base class for pytorch models
class multiHeadSelfAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()

    #output size must be evenly split among attention heads
    assert d_out % num_heads == 0 , "d_out must be divisible by num_heads"

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads

    #linear layers to project embeddings into query, key and value vectors
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    self.out_proj = nn.Linear(d_out, d_out) # linear layer to combine head outputs
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

  def forward(self, x):
    b, num_tokens, d_in= x.shape

    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    # reshape projections and tranpose for multi-head computation
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    attn_scores = queries @ keys.transpose(2, 3) # computing the dot product attention scores
    attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # apply mask to prevent future from being attended to

    # getting probability on importancet
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    #compute weighted sum of values and combine heads into original dim
    context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
    context_vec = self.out_proj(context_vec)
    return context_vec
  
class LayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, x):
      return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) *
        (x + 0.044715 * torch.pow(x, 3))
      ))


class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
      GELU(),
      nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
    )

  def forward(self, x):
    return self.layers(x)


class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.att = multiHeadSelfAttention(
        d_in=cfg["emb_dim"],
        d_out=cfg["emb_dim"],
        context_length=cfg["context_length"],
        num_heads=cfg["n_heads"], 
        dropout=cfg["drop_rate"],
        qkv_bias=cfg["qkv_bias"])
    self.ff = FeedForward(cfg)
    self.norm1 = LayerNorm(cfg["emb_dim"])
    self.norm2 = LayerNorm(cfg["emb_dim"])
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

  def forward(self, x):
    #shortcut connection for attention block
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)  
    x = self.drop_shortcut(x)
    x = x + shortcut  

    #shortcut connection for feed forward block
    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut  

    return x

class GPTModel(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"]) #prevent overfitting

    self.trf_blocks = nn.Sequential(
      *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    )

    self.final_norm = LayerNorm(cfg["emb_dim"]) #normalize output of last transformer block
    self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False) #probability distribution on next token

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape

    tok_embeddings = self.tok_emb(in_idx) # token into embedding
    pos_embeddings = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) # position embeddings for each token position

    x= tok_embeddings + pos_embeddings
    x= self.drop_emb(x) # dropout for regularization
    x= self.trf_blocks(x) # pass through stack of transformer blocks
    x= self.final_norm(x) # normalize final hidden states
    
    logits = self.out_head(x)
    return logits

def clac_loss_batch(input_batch, target_batch, model, device):
  #move data for efficient computation
  input_batch, target_batch = input_batch.to(device), target_batch.to(device) # tensors of tokenID's
  logits = model(input_batch)
  loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
  return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
  total_loss = 0.
  if len(data_loader) == 0:
    return float("nan")
  elif num_batches is None:
    num_batches = len(data_loader)
  else:
    num_batches = min(num_batches, len(data_loader))

  for i, (input_batch, target_batch) in enumerate(data_loader):
    if i< num_batches:
      loss = calc_loss_batch(input_batch, target_batch, model, device)
      total_loss += loss.item()
    else:
      break
  return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
  model.eval()
  with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def text_to_tokenizer(text, tokenizer):
  encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
  encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
  return encoded_tensor

def tokenizer_to_text(token_ids, tokenizer):
  flat = token_ids.squeeze(0)
  return tokenizer.decode(flat.tolist())

def generate_and_print_sample(model, tokenizer, device, start_context):
  model.eval()
  context_size = model.pos_emb.weight.shape[0]
  encoded = text_to_token_ids(start_context, tokenizer).to(device)
  with torch.no_grad():
    token_ids = generate_text_simple(
      model=model, idx=encoded,
      max_new_tokens=50, context_size=context_size
    )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
  model.train()

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
  fig, ax1 = plt.subplots(figsize=(5, 3))

  ax1.plot(epochs_seen, train_losses, label="Training loss")
  ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
  ax1.set_xlabel("Epochs")
  ax1.set_ylabel("Loss")
  ax1.legend(loc="upper right")
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

  ax2 = ax1.twiny()  # secondary x-axis sharing the same y-axis
  ax2.plot(tokens_seen, train_losses, alpha=0)  # invisible for alignment
  ax2.set_xlabel("Tokens seen")

  fig.tight_layout()
  plt.savefig("loss-plot.pdf")
  plt.show()


def generate_text_simple(model, idx, max_new_tokens, context_size):
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]  # limit to last `context_size` tokens
    with torch.no_grad():
      logits = model(idx_cond)
    logits = logits[:, -1, :]  # focus on the last token's logits
    probas = torch.softmax(logits, dim=-1)
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    idx = torch.cat((idx, idx_next), dim=1)  # append the predicted token
  return idx





    






