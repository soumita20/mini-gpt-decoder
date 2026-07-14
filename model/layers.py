import torch
import torch.nn as nn
from config import ModelConfig
from attention import CausalMultiHeadAttention

class RMSNorm(nn.Module):
    """
    Root mean square normalization. It is a computationally cheaper alternative to LayerNorm which omits the mean-centering step.
    """
    def __init__(self, dim:int, eps:float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #Calculate variance across the channel dimension without subtracting the mean
        variance = x.pow(2).mean(-1, keepdim=True)
        #Normalize and scale by the learnable weight parameter gamma
        return x*torch.rsqrt(variance+self.eps)* self.weight
    
class MLP(nn.Module):
    """
    Modern SwiGLU (Swish Gated Linear Unit) Feed forward network.
    Process token representations independently across a higher-dimensional space. 
    """
    def __init__(self, config:ModelConfig):
        super().__init__()
        #Calculating hidden layer dimensional scaling
        hidden_dim = int(2*(4*config.embedding_dim/3)/2)
        hidden_dim = ((hidden_dim+63)//64)*64

        #Gate(w1) and Up(w3) projections computed in parallel
        self.w1 = nn.Linear(config.embedding_dim, hidden_dim, bias = config.bias)
        self.w3 = nn.Linear(config.embedding_dim, hidden_dim, bias = config.bias)
        self.w2 = nn.Linear(config.embedding_dim, hidden_dim, bias = config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        #SwiGLU gating mechanism: SiLU (W1(x))*W3(x)
        return self.dropout(self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x)))
    
class Block(nn.Module):
    """
    The fundamental transformer decoder block. Combines Causal Self-Attention and SwiGLU MLP with pre-normalization residual streams.
    """
    def __init__(self, config:ModelConfig):
        super().__init__()
        #Pre-attention normalization
        self.rms_1 = RMSNorm(config.embedding_dim)
        self.attention = CausalMultiHeadAttention(config)
        #Pre-MLP normalization
        self.rms_2 = RMSNorm(config.embedding_dim)
        self.mlp = MLP(config)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        x shape : (batch, sequence_length, channel)
        Applies pre-RMSNorm residual connections to stabilize deep gradient flow.
        """
        #1. Attention residual block (Communication)
        x = x + self.attention(self.rms_1(x))

        #2. MLP Residual block(Computation)
        x = x + self.mlp(self.rms_2(x))

        return x

    
