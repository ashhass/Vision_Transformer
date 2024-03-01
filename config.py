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
        "in_channels": IN_CHANNELS,
        "mlp_ratio": MLP_RATIO,
        "qkv_bias": QKV_BIAS 
    },
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "load_model": LOAD_MODEL,
    "checkpoint": CHECKPOINT,
    "train_dir": TRAIN_DIR,
    "val_dir": VAL_DIR,
    "train_transform": TRAIN_TRANSFORM,
    "val_transform": VAL_TRANSFORM  
}
