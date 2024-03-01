from libs import *
from vit import ViT
from config import config
from train import Trainer


model = ViT(config['model_config'])
trainer = Trainer(config)
loss = trainer(model) 

print(loss)