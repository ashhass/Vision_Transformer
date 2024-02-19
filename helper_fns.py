


def load_checkpoint(checkpoint, model):
    print(f'== LOADING CHECKPOINT ==')
    model.load_state_dict(checkpoint["state_dict"]) 