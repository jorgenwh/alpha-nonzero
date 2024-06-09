import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary
from constants import D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE, BLOCK_SIZE, BATCH_SIZE, NUM_BINS


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.c_attn = nn.Linear(d_model, 3*d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = k.reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = k.reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)

        att = torch.matmul(q, k.transpose(2, 3))
        att = att / (k.shape[-1] ** 0.5)
        att = F.softmax(att, dim=3)
        att = torch.matmul(att, v)

        att = att.transpose(1, 2).reshape(B, L, D)

        att = self.c_proj(att)
        return att


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4*d_model)
        self.fc2 = nn.Linear(4*d_model, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model=d_model, n_heads=n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model)

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = residual + self.attn(x)

        residual = x
        x = self.ln2(x)
        x = residual + self.ff(x)
        return x


class Transformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 block_size,
                 output_size,
                 d_model,
                 n_heads,
                 n_layers
    ):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(block_size, d_model)

        self.blocks = nn.Sequential(
            *[Block(
                d_model=d_model,
                n_heads=n_heads
            ) for _ in range(n_layers)]
        )

        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        _, L = x.shape

        pos = torch.arange(0, L, dtype=torch.long, device=x.device).reshape(1, -1)
        pos_emb = self.positional_encoding(pos)
        tok_emb = self.input_embedding(x)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        x = self.fc(x)

        return x[:,-1,:]


if __name__ == '__main__':
    print(f"VOCAB_SIZE={VOCAB_SIZE}\nBLOCK_SIZE={BLOCK_SIZE}\nBATCH_SIZE={BATCH_SIZE}\nNUM_BINS={NUM_BINS}")
    x = torch.randint(VOCAB_SIZE, size=(BATCH_SIZE, BLOCK_SIZE))
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        output_size=NUM_BINS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS
    )

    print("Forwarding input of shape: " + str(list(x.shape)))
    y = model(x)
    print("Forward pass successful")

    print(summary(model, x))
