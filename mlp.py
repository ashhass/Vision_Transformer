from libs import torch, nn


class MLP(nn.Module):
    '''
        Input: tensor of shape (batch_size, num_patches + 1(+1 for the cls token), in_features)
        Output: tensor of shape (batch_size, num_patches + 1(+1 for the cls token), out_features)
    '''

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.dropout = nn.Dropout(p=p)

    
    def forward(self, x):    
        x = self.fc1(x)         # project input features to intermediate lower dimensional features
        x = self.act(x)         # inject some non-linearity into the equation
        x = self.dropout(x)     # drop some of the neurons to mitigate the risk of overfitting
        x = self.fc2(x)         # project the intermediate features to the final feature representation
        x = self.dropout(x)     # drop some more neurons to avoid overfitting

        return x    