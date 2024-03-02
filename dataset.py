from libs import *
from param import *

class CUB_Dataset():

    def __init__(self, dataset, transforms=None):
        super().__init__()
        image_list = []
        for root, dirs, files in os.walk(dataset):
            image_list += files if self.isImage(files) else []

        self.transform = transforms
        self.image_list = image_list
        self.dataset = dataset


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        image_idx = self.image_list[idx]
        for folder in os.listdir(f'{self.dataset}/images'):
            if folder[folder.find('.') + 1:] in image_idx:
                folder_idx = folder
        
        image = np.array(Image.open(f'{self.dataset}/images/{folder_idx}/{image_idx}'), dtype=np.int32)
        target = int(folder_idx[:folder_idx.find('.')])

        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, target


    def isImage(self, file):
        image_extension = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
        return any(file[0].endswith(extension) for extension in image_extension) 