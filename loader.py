from libs import *
from param import *

class CUB_Dataset():

    def __init__(self):
        super().__init__()
        imageToFolder, folderToName = {}, {}
        path = f'{dataset_folder}/images'
        for index, (root, dirs, files) in enumerate(os.walk(path)):
            imageToFolder[index] = [imageToFolder[index - 1][1], len(files) + imageToFolder[index - 1][1]] if index != 0 else [0, len(files)]
            folderToName[index] = root

        self.imageToFolder = imageToFolder
        self.folderToName = folderToName

        # print(self.imageToFolder)


    def __len__(self):
        return self.imageToFolder[200][1] - 1

    def __getitem__(self, idx):
        for keys, (low, high) in self.imageToFolder.items():
            if low <= idx <= high:
                folder = keys
                index = idx - low


        path = self.folderToName[folder]
        folderName = path[path.rfind('/') + 1 : ] 
        # print(folderName)
        
        image_path = f'{dataset_folder}/images/{folderName}'
        image_str = os.listdir(image_path)[index]

        image = np.array(Image.open(os.path.join(image_path, image_str)))

dataset = CUB_Dataset()
dataset.__getitem__(0)    
dataset.__getitem__(0)    
dataset.__getitem__(1)   
dataset.__getitem__(2)  