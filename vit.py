from libs import *
from param import *
from block import Block
from patch_embed import Patch_Embedding

class ViT(nn.Module):

    def __init__(self, config):
        super().__init__()

        num_patches = config['num_patches']
        embed_dim = config['embed_dim']
        img_size = config['img_size']
        num_classes = config['num_classes']
        patch_size = config['patch_size']
        num_encoders = config['num_encoders']
        num_heads = config['num_heads']
        hidden_dim = config['hidden_dim']
        dropout = config['dropout']
        in_channels = config['in_channels']
        mlp_ratio = config['mlp_ratio']
        qkv_bias = config['qkv_bias']
        
        self.embedding_block = Patch_Embedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        self.encoder_blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                p=dropout,
                attn_p=dropout
            )
            for _ in range(num_encoders)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_head = nn.Linear(in_features=embed_dim, out_features=num_classes)

    def forward(self, x):
        x = self.embedding_block(x)
        for block in self.encoder_blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.mlp_head(x[:, 0, :])
        
        return x 