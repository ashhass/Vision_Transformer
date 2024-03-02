from libs import *
from vit import ViT
from config import config
from train import Trainer
import multiprocessing


def main():
    model = ViT(config['model_config'])
    trainer = Trainer(config)
    loss = trainer(model) 

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main() 