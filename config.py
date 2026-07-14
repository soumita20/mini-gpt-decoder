"""This file contains the list of hyperparameters used for the training of the model. 
   It also contains a validation check to ensure that the embedding dimensions are completely divisible by the number of attention heads.
   I have frozen the dataclass so that it is immutable and cannot be modified in future by any other script or code."""

from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    #Character level vocabulary length
    vocab_size: int = 256
    #Context window length
    block_size: int = 256
    #Hidden dimension of the feedforward layers
    embedding_dim: int = 384
    #Number of attention heads(embed_dim//num_heads must be an integer)
    num_heads: int = 6
    #Depth of the decoder blocks
    num_layers: int = 6
    #Dropout rate for regularization
    dropout: float = 0.2
    #adding a bias term to the attention weights
    bias: bool = False

def __post_init__(self):
    #Ensuring that the embedding dimensions are completely divisible by the number of attention heads
    if self.embedding_dim % self.num_heads!=0:
        raise ValueError(f"Embedding dimension {self.embedding_dim} must be divisible by the number of attention heads {self.num_heads}.")