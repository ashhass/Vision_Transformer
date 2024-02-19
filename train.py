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
        self.num_epochs = config['NUM_EPOCHS']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.train_loader = train_loader
        self.val_loader = val_loader

        if self.config['load_model']:
            load_checkpoint(checkpoint=config['checkpoint'], model=self.model)


    def val_one_step(self, model, data):
        for k, v in data.items():
            data[k] = v.to(device)
        loss = model(**data)
        return loss


    def validate(self, model, dataloader):
        model.eval()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_idx, data in enumerate(self.val_loader):
                with torch.no_grad():
                    loss = val_one_step(model, data)
                total_loss += loss
        
        return total_loss
    

    def train_one_step(self, model, data, optimizer):
        optimizer.zero_grad()
        for k, v in data.items():
            data[k] = v.to(device)
        loss = model(**data)
        loss.backward()
        optimizer.step()
        return loss 
    

    def forward(self, x):
        model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_index, data in enumerate(self.train_loader):
                loss = train_one_step(model, data, optimizer)
                total_loss += loss

        return total_loss