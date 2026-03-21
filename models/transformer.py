import torch
import torch.nn as nn


class TransformerLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size=384,
        num_heads=8,
        num_layers=6,
        ff_hidden_size=1024,
        dropout=0.1,
        max_len=500,
        pad_token_id=None
    ):

        super().__init__()

        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=pad_token_id
        )
        self.position_embedding = nn.Embedding(max_len, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=ff_hidden_size,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_size, vocab_size, bias=False)

        # weight tying
        self.fc.weight = self.token_embedding.weight


    def generate_causal_mask(self, size):

        mask = torch.triu(torch.ones(size, size), diagonal=1)

        mask = mask.masked_fill(mask == 1, float('-inf'))

        return mask


    def forward(self, x):

        _, seq_len = x.shape

        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        padding_mask = None

        if self.pad_token_id is not None:
            padding_mask = x.eq(self.pad_token_id)

        x = self.token_embedding(x) + self.position_embedding(positions)

        mask = self.generate_causal_mask(seq_len).to(x.device)

        x = self.transformer(x, mask=mask, src_key_padding_mask=padding_mask)

        logits = self.fc(x)

        return logits
