from libs import DataLoader
from dataset import CUB_Dataset

def load_checkpoint(checkpoint, model):
    print(f'== LOADING CHECKPOINT ==')
    model.load_state_dict(checkpoint["state_dict"]) 


def get_loaders(train_dir, val_dir, train_transform, val_transform, batch_size, num_workers):

    train_ds = CUB_Dataset(train_dir)
    val_ds = CUB_Dataset(val_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader 