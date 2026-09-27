import torch
import torch.nn as nn
from torch.nn import functional as F

# Self-Attention Head
# ------------------
class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, block_size, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out


# Multi-Head Attention
# --------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embedding_dim, block_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(embedding_dim, block_size, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


# Feed-Forward Network
# --------------------
class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)


# Transformer Block
# -----------------
class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.sa_heads = MultiHeadAttention(num_heads, head_size, embedding_dim, block_size)
        self.ffwd = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
