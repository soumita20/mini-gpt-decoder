import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig
from model.embedding import TransformerEmbedding
from model.layers import RMSNorm, Block

class Transformer(nn.Module):
    """
    The top level auto-regressive transformer decoder network. 
    Co-ordinates embeddings, stacked decoder blocks and the final language model head.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        #1. Step 1:  Convert raw tokens to continuous vectors
        self.transformer = nn.ModuleDict(dict(
            wte = TransformerEmbedding(config), 
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f = RMSNorm(config.embedding_dim)
        ))

        #2. Map hidden features to vocabulary probabilities. Omitting bias as it stabilizes final probability distributions
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias = False)

        #3. Weight tying between embedding matrix and output projections. Saves VRAM 
        self.transformer.wte.token_embedding_table.weight = self.lm_head.weight

        #Initialize all network parameters safely
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Standard deep learning normalization initialization. """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx:torch.Tensor, targets: torch.Tensor = None)->tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx shape: (batch, channels) - Historical token inputs
        targets shape: (batch, channels) - Shifted ground truth tokens we want to predict
        """
        batch, channel = idx.shape
        assert channel <= self.config.block_size, f"Cannot forward sequence lengtth of {channel}, block size is {self.config.block_size}"

        #Pass through core structural layers

        #Embeddings
        x = self.transformer.wte(idx)

        #Sequentially route context through deep blocks
        for block in self.transformer.h:
            x = block(x)

        #Final normalization layer
        x = self.transformer.ln_f(x)

        #Calculate logits - raw predictions across vocabulary
        #Shape mutation: (batch, sequence_length, channel) -> (batch, sequence_length, vocab_size)
        logits = self.lm_head(x)

        #If in training mode, calculate loss
        loss = None
        if targets is not None:
            #Flatten tensors to fit Pytorch's cross entropy specifications
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx:torch.Tensor, max_new_tokens: int, temperature:float=1.0, top_k:int = None) -> torch.Tensor:
        """
        Auto-regressive generation loop. Takes a context prompt and predicts subsequent tokens.
        """
        for _ in range(max_new_tokens):
            #Crop current context if it exceeds our precomputed block_size limit
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            #Forward pass to fetch logits for absolute final token in the sequence
            logits, _ = self(idx_cond)

            #Isolate last step and scale by creativity
            logits = logits[:,-1,:]/temperature

            # Optional: Top-K sampling to prune low-probability noise tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply Softmax to get concrete probability distributions
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution to pick the next token ID
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the newly minted token onto our running stream
            idx = torch.cat((idx, next_token), dim=1)
            
        return idx



        