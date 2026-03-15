import torch
import torch.nn as nn


class TransformerLanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_size=256, num_heads=8, num_layers=4, max_len=256):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):

        seq_len = x.size(1)

        positions = torch.arange(0, seq_len).unsqueeze(0).to(x.device)

        x = self.token_embedding(x) + self.position_embedding(positions)

        x = self.transformer(x)

        logits = self.fc(x)

        return logits