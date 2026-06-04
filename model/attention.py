import math
import torch
#importing neural network modules
import torch.nn as nn
#importing functional API for activation functions and other operations
import torch.nn.functional as F
from config import ModelConfig

class RotaryEmbedding(nn.Module):
    def __init__(self,dim:int,max_seq_length:int = 256, base:float=10000.0):
        """dim: The specific dimension of the attention head
           max_seq_length: The maximum context length that the model uses.
           base: The theta value for rotating the Q and K vectors of a token."""
        #calling the constructor of the parent nn.Module class to initialize everything properly.
        super.__init__()
        self.dim = dim
        """torch.arange(0,dim,2) -> creates a tensor from 0 to dim with a step of 2.
           The result is then divided by dim to normalize the values between 0 and 1.
           The base is raised to the power of the normalized result to create geometric angles for the rotary embeddings."""
        inv_freq = 1.0/(base**(torch.arange(0,dim,2).float()/dim))
        #Saving the inv_freq as a buffer tensor in the module since it is not a learnable parameter.
        self.register_buffer("inv_freq",inv_freq,persistent=False)
        #Creating a 1D tensor of positions from 0 to max_seq_length-1.
        t = torch.arange(max_seq_length, dtype=torch.float)
        #Calculating the degree for the rotary embeddings by calculating the outer product of the above 2 tensors. 
        #The result is a matrix where each element (i,j) is the product of the i-th position and the j-th inverse frequency.
        freqs = torch.outer(t, self.inv_freq)
        """The freqs tensor is then duplicated along the last dimension to create a tensor 
        of shape (max_seq_length, dim) where each position has both sine and cosine components for the rotary embeddings."""
        emb = torch.cat((freqs, freqs), dim=-1)
        """All the precomputed sine and cosine values are stored as buffers in the module.
        These will be used during the forward pass to apply rotary embeddings to the Q and 
        K vectors dot product."""
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Here, the input tensor is split into 2 halves, 
            first half is negated,
            second half is unchanged,
            then they are concatenated back together along the last dimension 
            to create the rotated version of the tensor."""
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Here I will return the slice of precomputed sine and cosine tensors 
           matching the current sequence length.
           Output of this method will be multiplied to the dot product of Q and K vectors."""
        return self.cos_cached[..., :seq_len, :], self.sin_cached[..., :seq_len, :]