import os
import cv2

class DataLoader:
    def __init__(self, path, mode):
        """DataLoader Initialization

        Args:
            path (root path): Path to folder containing all classes folders
            mode (str): 'train' or 'test'
        """
        self.path = path
        self.mode = mode
        self.paths = []
        self.labels = []
        self.classes = []
        
        if os.path.exists(path):
            self.parse_data()
            
    
    def parse_data(self):
        """Load data from path
        """
        self.classes = os.listdir(self.path)
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith('.jpg'):
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
            label = self.labels[self.idx]
            self.idx += 1
            return img, label
        else:
            raise StopIteration
    
    
    
    