import os

from utils.dataloader import DataLoader

if __name__ == '__main__':
    mode = 'train'
    path = os.path.join('../Data/Challenge1/', mode)
    dataloader = DataLoader(path, mode)
    for img, label in dataloader:
        print(img.shape, label)