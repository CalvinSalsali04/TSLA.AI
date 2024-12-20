import torch
from llm import GPTModel  # Ensure this imports your model
import tiktoken  # Ensure tiktoken is installed

def load_model(cfg, checkpoint_path, device):
  # Initialize the model with the same config
  model = GPTModel(cfg).to(device)
  model.load_state_dict(torch.load(checkpoint_path, map_location=device))
  model.eval()
  return model

def generate_text(model, tokenizer, device, prompt, max_length=50):
  model.eval()
  with torch.no_grad():
    # Tokenize the prompt and convert to a PyTorch tensor
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)

    for _ in range(max_length):
      logits = model(input_ids)
      next_token_logits = logits[:, -1, :]  # Get logits for the last token
      next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # Choose the most probable token
      input_ids = torch.cat([input_ids, next_token], dim=-1)  # Append the new token

    # Decode the generated tokens back to text
    return tokenizer.decode(input_ids[0].tolist())


if __name__ == "__main__":
  # Configuration should match what you used during training
  cfg = {
    "vocab_size": 50257,
    "emb_dim": 768,
    "context_length": 256,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
  }
  checkpoint_path = "model_epoch_10.pth"  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Load tokenizer
  tokenizer = tiktoken.get_encoding("gpt2")  # Ensure you use the same tokenizer

  # Load model
  model = load_model(cfg, checkpoint_path, device)

  # Initialize conversation history
  history = ""

  # Chat loop
  print("Start chatting with your LLM! Type 'exit' to end.")
  while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
      print("Goodbye!")
      break

    # Append user input to history
    history += f"You: {user_input}\n"

    # Generate response based on the entire conversation history
    response = generate_text(model, tokenizer, device, history, max_length=50)

    # Append model response to history
    history += f"LLM: {response}\n"

    # Print the model's response
    print(f"LLM: {response}")
