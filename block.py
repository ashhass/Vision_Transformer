from libs import torch, nn
from attention import Attention

class Block(nn.Module):

    '''
        Parameters:
            1. num_heads: how many attention heads to use
            2. dim: embedding dimension
            3. mlp_ratio: determines the hidden dimension size of the MLP module with respect to dim
            4. qkv_bias: Boolean of whether to include bias to the query, key and value projections
            5. p, attn_p: dropout probability of mlp and attention layers respectively

        Attributes:
            1. norm: 2 separate layer normalization layers both having their own parameters
            2. attn: attention layer
            3. mlp: linear layer
    '''

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim=dim, 
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_prob=p,
            attn_p=attn_p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x