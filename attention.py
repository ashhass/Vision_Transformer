from libs import torch, nn


class Attention(nn.Module):

    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_prob=0., proj_prob=0.):
        super().__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_prob)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_prob)


    def forward(self, x):
        num_samples, num_tokens, dim = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(num_samples, num_tokens, 3, self.num_heads, self.head_dim) 
        qkv = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)

        dot_prod = (q @ k_t) * self.scale
        attn = dot_prod.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(1, 2)
        weighted_avg = weighted_avg.flatten(2) 

        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x