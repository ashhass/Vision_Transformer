from param import *

config = {
    "model_config": {
        "num_patches": NUM_PATCHES,
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "patch_size": PATCH_SIZE,
        "embed_dim": EMBED_DIM,
        "num_encoders": NUM_ENCODERS,
        "num_heads": NUM_HEADS,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "activation": ACTIVATION,
        "in_channels": IN_CHANNELS
    },
    "NUM_EPOCHS": NUM_EPOCHS,
    "load_model": LOAD_MODEL,
    "checkpoint": CHECKPOINT  
}
