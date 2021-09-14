import os
import numpy as np
from glob import glob,iglob
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageFolder(Dataset):
    def __init__(self,dir_name="train"):
        image_dir =  "DIV2K_{}_HR".format(dir_name)
        root_dir = "{}/{}".format(image_dir,image_dir)
        self.data = glob(os.path.join(root_dir,"*.png"))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        image = np.array(Image.open(self.data[index]))
        image = config.both_transforms(image=image)['image']
        high_res = config.highres_transform(image=image)['image']
        low_res = config.lowres_transform(image=image)['image']
        return low_res,high_res
    
def test():
    dataset = ImageFolder(dir_name="train")
    loader = DataLoader(dataset,batch_size=1,num_workers=0)
    for low_res,high_res in loader:
        print(low_res.shape)
        print(high_res.shape)
        print("=============")