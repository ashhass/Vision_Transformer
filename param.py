from libs import torch

LEARNING_RATE = 1e-4
NUM_CLASSES = 10
PATCH_SIZE = 4
IMG_SIZE = 28
IN_CHANNELS = 1
NUM_HEADS = 8
DROPOUT = 0.001
HIDDEN_DIM = 768
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION = 'gelu'
NUM_ENCODERS = 4
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS
NUM_PATCHES = (IMG_SIZE  // PATCH_SIZE) ** 2
NUM_EPOCHS = 4
LOAD_MODEL = False
CHECKPOINT = ''
BATCH_SIZE = 4
NUM_WORKERS = 4
TRAIN_DIR = '/Users/aydasultan/Documents/CUB_Dataset/CUB_200_2011' 
VAL_DIR = '/Users/aydasultan/Documents/CUB_Dataset/CUB_200_2011' 
dataset_folder = '/Users/aydasultan/Documents/CUB_Dataset/CUB_200_2011' 


device = "cuda" if torch.cuda.is_available() else "cpu"