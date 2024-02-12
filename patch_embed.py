from libs import *
from param import *

class Patch_Embedding(nn.Module):


    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = embed_dim,
                kernel_size = patch_size,
                stride = patch_size
            ),
            nn.Flatten(2))

        self.cls_token = nn.Parameter(torch.randn(size=(1, in_channels, embed_dim)), requires_grad=True)
        self.positional_embedding = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True) # add 1 for the cls token
        self.dropout = nn.Dropout(p = dropout)
    
    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patch_embed(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.dropout(x)

        return x


model = Patch_Embedding(EMBED_DIM, PATCH_SIZE, NUM_PATCHES, DROPOUT, IN_CHANNELS)
x =  torch.rand(512, 1, 28, 28)
print(model(x).shape)