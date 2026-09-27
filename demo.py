import torch
import torch.nn as nn
import torch.nn.functional as F 
import random

from transformer_block import Block


# Corpus Data
corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Aarohi",
    "the roads of Delhi are busy",
    "it is raining in Mumbai",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "holi is my favorite festival",
    "diwali brings lights and sweets",
    "india won the cricket match"
]

corpus = [ s + " <END>" for s in corpus ]
text = " ".join(corpus)
print(text)

words = list(set(text.split()))
vocab_size = len(words)
print(words)
print("-------------------------------------------------------------------------------")
word2idx = { w : i for i,w in enumerate(words)}
idx2word = { i : w for w,i in word2idx.items()}
print(word2idx)
print("--------------------------------------------------------------------------------")
data = torch.tensor([word2idx[w] for w in text.split()], dtype = torch.long)
print(data)

print("converting the data in batches --------------------------------------------------")

block_size = 6
embedding_dim = 32      
n_heads = 2
n_layers = 2
lr = 1e-3
epochs = 1500


def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 62-6=55 
    #[12,32,44,....55]
    #[12,13,14,15..]
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) # (42,32)
        #gives the position value to each word
        self.position_embedding = nn.Embedding(block_size, embedding_dim) # (6,32)
        self.blocks = nn.Sequential(*[Block(embedding_dim=embedding_dim, num_heads=n_heads, block_size=block_size) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss



    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop current context sequence to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            
            # Get model predictions (forward pass)
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step prediction logits
            logits = logits[:, -1, :]
            
            # Convert raw logits to probabilities using Softmax
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token index from probability distribution
            next_idx = torch.multinomial(probs, 1)
            
            # Append sampled index to running sequence
            idx = torch.cat((idx, next_idx), dim=1)
            
        return idx


model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")



# Define starting seed prompt token ("hello")
context = torch.tensor([[word2idx["the"]]], dtype=torch.long)

# Generate up to 15 new tokens
out = model.generate(context, max_new_tokens=5)

# Decode generated token indices back to human-readable text
print(" ".join(idx2word[int(i)] for i in out[0]))
