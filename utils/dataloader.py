import os
import cv2
import random
from collections import Counter

import numpy as np
from sklearn.utils import resample

class DataLoader:
    def __init__(self, path, mode, transforms=None, shuffle=False, ignore_folders=[], max_samples=None, balance=False, classes=None, mask=False):
        """DataLoader Initialization

        Args:
            path (root path): Path to folder containing all classes folders
            mode (str): 'train' or 'test'
            transforms (callable, optional): Optional transforms to be applied on a sample.
            shuffle (bool, optional): Shuffle data. Defaults to False.
            ignore_folders (list, optional): List of folders to ignore. Defaults to [].
            max_samples (int, optional): Maximum number of samples to load. Defaults to None.
            balance (str, optional): Balance data w.r.t max_samples. Defaults to None.
            classes (list, optional): List of classes. Defaults to None.
            mask (bool, optional): If True, load mask images. Defaults to False.
        """
        self.path = os.path.join(path, mode)
        self.mode = mode
        self.paths = []
        self.labels = []
        self.classes = classes
        self.transforms = transforms
        self.ignore_folders = ignore_folders
        self.balance = balance
        self.max_samples = max_samples
        self.mask = mask
        
        if os.path.exists(path):
            self.parse_data()
            
            ## Shuffle data
            if shuffle:
                indices = list(range(len(self)))
                random.shuffle(indices)
                self.paths = [self.paths[i] for i in indices]
                self.labels = [self.labels[i] for i in indices]
                
            ## Balance data
            if balance:
                self.balance_data()
            
            ## Limit samples
            if max_samples is not None and max_samples < len(self):
                self.paths = self.paths[:max_samples]
                self.labels = self.labels[:max_samples]
            
            
    
    def parse_data(self):
        """Load data from path
        """                
        for root, dirs, files in os.walk(self.path):                
            for file in files:
                if file.endswith('.jpg') and os.path.basename(root) not in self.ignore_folders:
                    self.paths.append(os.path.join(root, file))
                    self.labels.append(self.classes.index(os.path.basename(root)))
        
        
    def balance_data(self):
        """Balance the data across classes using the specified sampling method."""
        label_counts = Counter(self.labels)
        max_samples_per_class = self.max_samples // len(self.classes)

        balanced_paths = []
        balanced_labels = []

        # Balance data class by class
        for class_ix in range(len(self.classes)):
            # Get all samples for this class
            class_paths = [self.paths[i] for i in range(len(self.paths)) if self.labels[i] == class_ix]
            class_labels = [self.labels[i] for i in range(len(self.labels)) if self.labels[i] == class_ix]

            # If there are more samples than max_samples_per_class, downsample
            if label_counts[class_ix] > max_samples_per_class:
                class_paths, class_labels = resample(
                    class_paths, class_labels, n_samples=max_samples_per_class, random_state=42, replace=False
                )

            # If there are fewer samples than max_samples_per_class, upsample
            elif label_counts[class_ix] < max_samples_per_class:
                class_paths, class_labels = resample(
                    class_paths, class_labels, n_samples=max_samples_per_class, random_state=42, replace=True
                )

            # Append the resampled paths and labels to the balanced lists
            balanced_paths.extend(class_paths)
            balanced_labels.extend(class_labels)

        # Update self.paths and self.labels with the balanced data
        self.paths = balanced_paths
        self.labels = balanced_labels
        
                
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
            mask = cv2.imread(self.paths[self.idx].replace(self.mode, 'mask/' + self.mode), cv2.IMREAD_GRAYSCALE) if self.mask else None
            
            # if mask is not None:
            #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     if len(contours) > 0:
            #         contour = max(contours, key=cv2.contourArea)
            #         x, y, w, h = cv2.boundingRect(contour)
            #         img = img[y:y+h, x:x+w]
                    
            
            im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            im_hist = cv2.calcHist([im_gray], [0], None, [256], [0, 256])
            
            if im_hist[0] > 0.1*img.shape[0]*img.shape[1]:
                ## crop the image corresponding to the biggest contour
                ot, otsu_mask = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(otsu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contour)
                    img = img[y:y+h, x:x+w]
            
            if self.transforms is not None:
                img = self.transforms(img)
            label = self.labels[self.idx]
            path =  self.paths[self.idx]
            self.idx += 1
            return img, label, mask, path
        else:
            raise StopIteration
    
    
    
    