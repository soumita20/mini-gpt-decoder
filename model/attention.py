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
        """The first half of the tensor is paired with the 2nd half to create the pair of co-ordinates for the rotary embedding.
            Here, the input tensor is split into 2 halves, 
            first half is negated,
            second half is unchanged,
            then they are concatenated back together along the last dimension 
            to create the rotated version of the tensor.
            """
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)
    
def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Here I will return the slice of precomputed sine and cosine tensors 
        matching the current sequence length.
        Output of this method will be multiplied to the dot product of Q and K vectors."""
    return self.cos_cached[..., :seq_len, :], self.sin_cached[..., :seq_len, :]


# This method actually applies the rotary embedding to the input tensor.
def apply_rotary_embedding(self, x:torch.Tensor, cos_value:torch.Tensor, sin_value:torch.Tensor) -> torch.Tensor:
    return (x*cos_value) + (self.rotate_half(x*sin_value))
    
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads

        """Calculating the attention matrix"""
        self.attention_matrix = nn.Linear(config.embedding_dim, 3*config.embedding_dim, bias=config.bias)
        
        self.projection_layer = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias)

        """ Adding regularization layers to prevent overfitting """
        self.attention_droput_layer = nn.Dropout(config.dropout)
        self.residual_dropout_layer = nn.Dropout(config.dropout)

        """Adding the rotary embedding layer to implement rotary positional encodings."""
        self.rotary_embedding = RotaryEmbedding(dim = self.head_dim, max_seq_length = config.block_size)

        """Adding the mask so that tokens cannot look at previous tokens"""
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch, sequence_length, channels = x.shape
        query, key, value = self.c_attn(x).split(self.embed_dim, dim=1)

        #Reshaping to multi-headed architecture
        query = query.view(batch, sequence_length, self.num_heads, self.head_dim).transpose(1,2)
        key = key.view(batch, sequence_length, self.num_heads, self.head_dim).transpose(1,2)
        value = value.view(batch, sequence_length, self.num_heads, self.head_dim).transpose(1,2)

        #Fetch cached sine and cosine rotations for current sequence length
        cos, sin = self.rotary_embedding(query, sequence_length=sequence_length)

        #Applying RoPE rotations directly to queries and keys
        query = apply_rotary_embedding(query, cos, sin)

        #Calculating the attention matrix
        attention_matrix = (query @ key.transpose(-2,-1)) * (1.0 /math.sqrt(self.head_dim))

        #Blind future tokens with negative infinity
        attention = attention.masked_fill(self.bias[:,:,:sequence_length,:sequence_length] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        attention = self.attention_droput_layer(attention)

        #Weighted sum over values
        weighted_sum = attention @ value

        #Compress the matrix into a single continuous vector
        weighted_sum = weighted_sum.transpose(1,2).contigiuous().view(batch,sequence_length,channels)

        return self.residual_dropout_layer(self.projection_layer(weighted_sum))

        



