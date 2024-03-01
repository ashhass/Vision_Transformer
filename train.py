from libs import torch, nn
from vit import ViT
from config import config as cfg
from helper_fns import get_loaders


class Trainer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ViT(config['model_config'])
        self.num_epochs = config['num_epochs']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        train_loader, val_loader = get_loaders(
            train_dir=cfg["train_dir"], val_dir=cfg["val_dir"], train_transform=cfg["train_transform"], 
            val_transform=cfg["val_transform"], batch_size=cfg["batch_size"], num_workers=cfg["num_workers"]
        )    
        self.train_loader = train_loader
        self.val_loader = val_loader

        if self.config['load_model']:
            load_checkpoint(checkpoint=config['checkpoint'], model=self.model)


    def val_one_step(self, model, data):
        for k, v in data.items():
            data[k] = v.to(device)
        
        inputs, target = data.items()
        predictions = model(inputs)
        
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
        
        inputs, target = data.items()
        predictions = model(inputs)
        
        loss = self.criterion(predictions, target)  
        loss.backward()
        optimizer.step()
        
        return loss 
    

    def forward(self, model):
        model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_index, data in enumerate(self.train_loader):
                with torch.cuda.amp.autocast:
                    loss = train_one_step(model, data, self.optimizer)
                total_loss += loss

        return total_loss   