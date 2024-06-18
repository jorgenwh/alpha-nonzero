import torch
import torch.nn as nn
import torch.nn.functional as F
from ..constants import D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE, BLOCK_SIZE, POLICY_SIZE, BATCH_SIZE, DEVICE


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(D_MODEL, 3*D_MODEL)
        self.c_proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        B, L, D = x.shape

        q, k, v = self.c_attn(x).split(D_MODEL, dim=2)
        q = k.reshape(B, L, N_HEADS, D // N_HEADS).transpose(1, 2)
        k = k.reshape(B, L, N_HEADS, D // N_HEADS).transpose(1, 2)
        v = k.reshape(B, L, N_HEADS, D // N_HEADS).transpose(1, 2)

        att = torch.matmul(q, k.transpose(2, 3))
        att = att / (k.shape[-1] ** 0.5)
        att = F.softmax(att, dim=3)
        att = torch.matmul(att, v)

        att = att.transpose(1, 2).reshape(B, L, D)

        att = self.c_proj(att)
        return att


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, 4*D_MODEL)
        self.fc2 = nn.Linear(4*D_MODEL, D_MODEL)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = Attention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = FeedForward()

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = residual + self.attn(x)

        residual = x
        x = self.ln2(x)
        x = residual + self.ff(x)
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.positional_encoding = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.positional_vector = torch.arange(0, BLOCK_SIZE, dtype=torch.int64, device=DEVICE).reshape(1, BLOCK_SIZE)

        self.blocks = nn.Sequential(
            *[Block() for _ in range(N_LAYERS)]
        )

        self.ln = nn.LayerNorm(D_MODEL)

        self.v_fc = nn.Linear(D_MODEL, 256)
        self.v = nn.Linear(256, 1)

        self.pi_fc = nn.Linear(D_MODEL, 256)
        self.pi = nn.Linear(256, POLICY_SIZE)

    def forward(self, x):
        B, _ = x.shape

        # get positional and token embeddings
        pos_emb = self.positional_encoding(self.positional_vector)
        tok_emb = self.input_embedding(x)

        # forward pass through the transformer
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)

        # get the last embedded prediction
        x = x[:,-1,:].reshape(B, D_MODEL)

        # forward pass for the value head
        v = self.v_fc(x)
        v = F.relu(v)
        v = self.v(v)
        v = torch.tanh(v)

        # forward pass for the policy head
        pi = self.pi_fc(x)
        pi = F.relu(pi)
        pi = self.pi(pi)

        return pi, v
