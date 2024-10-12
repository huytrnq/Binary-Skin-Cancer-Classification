import os
import cv2
from random import shuffle

class DataLoader:
    def __init__(self, path, mode, transforms=None, shuffle=False, ignore_folders=[]):
        """DataLoader Initialization

        Args:
            path (root path): Path to folder containing all classes folders
            mode (str): 'train' or 'test'
            transforms (callable, optional): Optional transforms to be applied on a sample.
            shuffle (bool, optional): Shuffle data. Defaults to False.
            ignore_folders (list, optional): List of folders to ignore. Defaults to [].
        """
        self.path = os.path.join(path, mode)
        self.mode = mode
        self.paths = []
        self.labels = []
        self.classes = []
        self.transforms = transforms
        self.ignore_folders = ignore_folders
        
        if os.path.exists(path):
            self.parse_data()
            
        if shuffle:
            indices = list(range(len(self)))
            shuffle(indices)
            self.paths = [self.paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            
    
    def parse_data(self):
        """Load data from path
        """
        self.classes = os.listdir(self.path)
        for folder in self.ignore_folders:
            if folder in self.classes:
                self.classes.remove(folder)
                
        for root, dirs, files in os.walk(self.path):                
            for file in files:
                if file.endswith('.jpg') and os.path.basename(os.path.dirname(root)) not in self.ignore_folders:
                    self.paths.append(os.path.join(root, file))
                    self.labels.append(self.classes.index(os.path.basename(root)))
        
    
    def __len__(self):
        """Get length of DataLoader

        Returns:
            int: Length of DataLoader
        """
        return len(self.paths)
    
    def __iter__(self):
        """Get iterator of DataLoader

        Returns:
            DataLoader: DataLoader iterator
        """
        self.idx = 0
        return self
    
    def __next__(self):
        """Get next data

        Returns:
            tuple: (image, label)
        """
        if self.idx < len(self):
            img = cv2.imread(self.paths[self.idx])
            if self.transforms is not None:
                img = self.transforms(img)
            label = self.labels[self.idx]
            path =  self.paths[self.idx]
            self.idx += 1
            return img, label, path
        else:
            raise StopIteration
    
    
    
    