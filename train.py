from libs import *
from vit import ViT
from config import config


'''
    What do I need:     
        1. Load the data
        2. Initialize the cost function and optimization algorithm
        3. Train the network on loaded data

''' 



class Trainer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ViT(**config['model_config'])
        self.NUM_EPOCHS = config['NUM_EPOCHS']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        if self.config['load_model']:
            load_checkpoint(checkpoint=config['checkpoint'], model=self.model)

    
    def forward(self, x):
        model.train()
        total_loss = 0

        for epoch in range(NUM_EPOCHS):
            pass
