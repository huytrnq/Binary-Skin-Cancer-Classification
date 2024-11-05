import os
from tqdm import tqdm

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score

class MultifeatureBoW:
    def __init__(self, vocab_size=100, descriptors=[], classifier=None):
        """
        Initializes the Bag of Words model for multiple feature types.

        Args:
            vocab_size (int): Number of visual words in the BoW codebook.
            descriptors (list): List of descriptor classes (e.g., LBPDescriptor, GLCMDescriptor, etc.).
            classifier (object): Classifier object to use for classification
        """
        self.vocab_size = vocab_size
        self.descriptors = descriptors
        self.kmeans_models = {desc.__class__.__name__: KMeans(n_clusters=vocab_size, random_state=0) for desc in descriptors}
        self.scalers = {desc.__class__.__name__: StandardScaler() for desc in descriptors}
        
        # Default to SVM if no classifier is provided
        self.classifier = classifier if classifier else SVC()

    def build_codebook(self, dataloader):
        """
        Builds the codebook using KMeans for all descriptors.
        
        Args:
            dataloader (DataLoader): DataLoader object containing the training data.
        """
        for descriptor in self.descriptors:
            desc_name = descriptor.__class__.__name__
            descriptors_list = []

            # Extract features using the current descriptor
            for image, label, mask, path in tqdm(dataloader, desc="Extracting Features for " + desc_name):
                features = descriptor.extract(image, mask)
                ## If the descriptor returns a tuple, take the first element
                if type(features) == tuple:
                    features = features[0]
                descriptors_list.append(features)

            # Stack, normalize, and fit KMeans for the current descriptor
            descriptors = np.vstack(descriptors_list)
            descriptors = self.scalers[desc_name].fit_transform(descriptors)
            self.kmeans_models[desc_name].fit(descriptors)

    def extract_bow_histogram(self, image):
        """
        Creates Bag of Words histograms for multiple feature types.
        
        Args:
            image (numpy array): Input image.
        
        Returns:
            histogram (numpy array): Concatenated BoW histogram for the image.
        """
        histograms = []
        
        # Extract features for each descriptor and create BoW histogram
        for descriptor in self.descriptors:
            desc_name = descriptor.__class__.__name__
            descriptors = descriptor.extract(image)
            if type(descriptors) == tuple:
                    descriptors = descriptors[0]
            descriptors = self.scalers[desc_name].transform(np.array(descriptors).reshape(1, -1))
            labels = self.kmeans_models[desc_name].predict(descriptors)
            hist, _ = np.histogram(labels, bins=self.vocab_size, range=(0, self.vocab_size))
            hist = hist.astype("float") / (np.sum(hist) + 1e-7)
            histograms.append(hist)
        
        # Concatenate histograms from all descriptors
        return np.concatenate(histograms)
    
    def transform_from_dataloader(self, dataloader):
        """
        Transforms images into their BoW histogram representations.
        
        Args:
            dataloader (DataLoader): DataLoader object containing the images to transform.
        
        Returns:
            histograms (numpy array): BoW histograms for each image.
            labels (numpy array): Labels for each image.
        """
        histograms = []
        labels = []
        
        for image, label, mask, path in tqdm(dataloader, desc="Transforming Images"):
            histograms.append(self.extract_bow_histogram(image))
            labels.append(label)
        
        return np.array(histograms), labels

    def transform(self, images):
        """
        Transforms images into their BoW histogram representations.
        
        Args:
            image_data (list): List of images to transform paths or numpy arrays.
        
        Returns:
            histograms (numpy array): BoW histograms for each image.
        """
        histograms = []
        
        for image in images:
            if isinstance(image, str):
                image = cv2.imread(image)
            histograms.append(self.extract_bow_histogram(image))
        
        return np.array(histograms)
    
    def fit_classifier(self, dataloader):
        """
        Fits the classifier using the BoW histograms.
        
        Args:
            dataloader (DataLoader): DataLoader object containing the training data.
        """
        print("Fitting classifier...")
        histograms, labels = self.transform_from_dataloader(dataloader)
        self.classifier.fit(histograms, labels)
        
    def predict(self, dataloader):
        """
        Predicts the labels for a given set of images using the trained classifier.
        
        Args:
            dataloader (DataLoader): DataLoader object that yields batches of images.
        
        Returns:
            predictions (numpy array): Predicted labels for the images.
        """
        print("Predicting...")
        histograms, _ = self.transform_from_dataloader(dataloader)  # We don't need the labels during inference
        return self.classifier.predict(histograms)
    
    def evaluate(self, dataloader, class_names=None):
        """
        Evaluates the classifier using the given DataLoader.
        
        Args:
            dataloader (DataLoader): DataLoader object containing the evaluation data.
        
        Returns:
            accuracy (float): Classification accuracy.
        """
        print("Evaluating...")
        predictions = self.predict(dataloader)
        labels = np.array([label for _, label, mask, _ in dataloader])
        return predictions, labels