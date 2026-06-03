import torch
import torch.nn as nn
from config import ModelConfig

class TransformerEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Embedding table created for storing the vocabulary embeddings.
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embd_dim)
        # Embedding table created for storing the positional encodings.
        self.position_embedding_table = nn.Embedding(config.block_size, config.embd_dim)
        # Dropout layer added for regularization to prevent overfitting.
        self.dropout = nn.Dropout(config.dropout)

    # The forward method takes in a tensor of token indices and returns the corresponding embeddings.
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # B ->Batch size, meaning the number of sentences to be processed at once, T->Sequence length, meaning the number of tokens in each sentence.
        B,T = idx.size()
        # Fetching the token embeddings from the vocabulary embedding table using the input tensor of token indices.
        # Output shape of token_emb is (B,T,config.emd_dim)
        x = self.token_embedding_table(idx)
        return self.dropout(x)