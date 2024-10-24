import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops


import cv2
import numpy as np
from skimage.feature import local_binary_pattern

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

class LBPDescriptor:
    def __init__(self, radius=1, n_points=8, grid_x=1, grid_y=1, visualize=False):
        """
        Initializes the LBPDescriptor with spatial grid parameters.

        Args:
            radius (int): Radius for the LBP.
            n_points (int): Number of circularly symmetric points considered for LBP.
            grid_x (int): Number of grids along the X-axis (width).
            grid_y (int): Number of grids along the Y-axis (height).
            visualize (bool): Whether to visualize the LBP image.
        """
        self.radius = radius
        self.n_points = n_points
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.visualize = visualize

    def extract(self, image, mask=None):
        """
        Extracts Local Binary Patterns (LBP) from the image, dividing it into grids.

        Args:
            image (numpy array): The image from which to extract LBP features.
            mask (numpy array): Optional mask to apply to the image.

        Returns:
            concatenated_hist (numpy array): Concatenated histogram of LBP features from all grids.
        """
        # Convert the image to grayscale if it's not already
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply mask to the image if provided
        if mask is not None:
            masked_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        else:
            masked_image = gray_image

        # Get the image size
        height, width = masked_image.shape
        
        # Calculate the size of each grid
        grid_height = height // self.grid_y
        grid_width = width // self.grid_x

        # Initialize the final concatenated histogram
        concatenated_hist = []

        # Initialize an empty image to store the final LBP representation
        if self.visualize:
            lbp_image = np.zeros_like(masked_image)
        else:
            lbp_image = None

        # Loop over the grids
        for i in range(self.grid_y):
            for j in range(self.grid_x):
                # Extract the sub-region (grid) from the image
                start_y = i * grid_height
                end_y = start_y + grid_height
                start_x = j * grid_width
                end_x = start_x + grid_width
                
                # Get the current grid
                grid = masked_image[start_y:end_y, start_x:end_x]
                
                # Compute LBP for the current grid
                lbp = local_binary_pattern(grid, self.n_points, self.radius, method="uniform")
                
                # Calculate LBP histogram for the current grid
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.n_points + 3), range=(0, self.n_points + 2))
                
                # Normalize the histogram
                lbp_hist = lbp_hist.astype("float")
                lbp_hist /= (lbp_hist.sum() + 1e-7)

                # Append the histogram to the concatenated histograms
                concatenated_hist.extend(lbp_hist)

                if self.visualize:
                    # Store the LBP result into the lbp_image
                    lbp_image[start_y:end_y, start_x:end_x] = lbp

        # Convert to numpy array
        return np.array(concatenated_hist), lbp_image



class GLCMDescriptor:
    def __init__(self, distances=[1], angles=[0], levels=8):
        """
        Initializes the GLCMDescriptor.

        Args:
            distances (list): Distances for GLCM computation.
            angles (list): Angles for GLCM computation.
            levels (int): Number of quantization levels for the GLCM.
        """
        self.distances = distances
        self.angles = angles
        self.levels = levels

    def extract(self, image, mask=None):
        """
        Extracts Gray Level Co-occurrence Matrix (GLCM) features from the image.

        Args:
            image (numpy array): The image from which to extract GLCM features.
            mask (numpy array): Optional mask to apply to the image.

        Returns:
            features (list): GLCM features including contrast, correlation, energy, and homogeneity.
        """
        # Convert the image to grayscale if it's not already
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply mask to the image if provided
        if mask is not None:
            masked_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        else:
            masked_image = gray_image

        # Quantize the image to a fixed number of gray levels (for GLCM)
        quantized_image = self.quantize_image(masked_image)

        # Compute GLCM
        glcm = graycomatrix(quantized_image, distances=self.distances, angles=self.angles, levels=self.levels,
                            symmetric=True, normed=True)

        # Extract GLCM features
        contrast = graycoprops(glcm, 'contrast').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        energy = graycoprops(glcm, 'energy').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()

        # Combine all GLCM features into a single list
        glcm_features = [contrast, correlation, energy, homogeneity]

        return glcm_features

    def quantize_image(self, image):
        """
        Quantizes the grayscale image into a specified number of levels.

        Args:
            image (numpy array): Grayscale image.

        Returns:
            quantized_image (numpy array): Quantized image.
        """
        quantized_image = np.floor(image / 256.0 * self.levels).astype(np.uint8)
        return quantized_image