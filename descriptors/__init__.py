import numpy as np
import cv2

import cv2
import numpy as np

class ColorDescriptor:
    def __init__(self, bins):
        """
        Initializes the color descriptor.
        
        Args:
            bins (tuple): The number of bins for each channel in the HSV color space.
        """
        self.bins = bins

    def extract(self, image):
        """
        Extracts a color histogram from an image in the HSV color space.

        Args:
            image (numpy array): The image from which to extract the color features.

        Returns:
            features (list): A flattened list of the histogram values normalized.
        """
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Initialize the color histogram
        features = []

        # Extract the color histogram from the entire image
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, self.bins, 
                            [0, 180, 0, 256, 0, 256])

        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()

        # Add the histogram to the feature vector
        features.extend(hist)

        return features

