import os
from Challenge1.utils.transforms import Composer, ObjectCentricCropping, HairRemoval

from utils.dataloader import DataLoader

if __name__ == '__main__':
    mode = 'train'
    path = os.path.join('../Data/Challenge1/', mode)
    dataloader = DataLoader(path, mode)
    for img, label in dataloader:
        print(img.shape, label)
        
    
    transform_composer = Composer([
        ObjectCentricCropping((128, 128)), 
        HairRemoval()])
    
    transformed_img = transform_composer(img)