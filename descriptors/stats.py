import cv2
import numpy as np
from scipy.stats import skew, kurtosis, entropy

class IntensityStatsGridDescriptor:
    """
    Calculates intensity-based statistical features (mean, std, skewness, kurtosis, entropy)
    within a grid layout over a specified grayscale region of interest (ROI).
    """
    
    def __init__(self, grid_x=4, grid_y=4):
        """
        Initializes the grid shape for statistical feature calculation.
        
        Parameters:
        - grid_x (int): Number of grid cells along the X-axis.
        - grid_y (int): Number of grid cells along the Y-axis.
        """
        self.grid_size = (grid_x, grid_y)

    def calculate_intensity_stats(self, intensity_values):
        """
        Calculates statistical features for the intensity values within a grid cell.

        Parameters:
        - intensity_values (array): Flattened 1D array of intensity values within the grid cell.

        Returns:
        - stats (list): List containing mean, std, skewness, kurtosis, and entropy.
        """
        mean_intensity = np.mean(intensity_values)
        std_intensity = np.std(intensity_values)
        skewness = skew(intensity_values)
        kurt = kurtosis(intensity_values)
        ent = entropy(np.histogram(intensity_values, bins=256, range=(0, 256))[0])

        return [mean_intensity, std_intensity, skewness, kurt, ent]
    
    def extract(self, img, mask=None):
        """
        Divides the grayscale ROI into grids and calculates intensity statistics for each cell.

        Parameters:
        - img (ndarray): image from which to extract features.
        - mask (ndarray): Optional mask to apply to the image.

        Returns:
        - feature_vector (list): Concatenated list of intensity-based statistical features for each grid cell.
        """
        
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ## Apply mask to the image if provided
        if mask is not None:
            gray_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        else:
            gray_image = gray_image
        
        h, w = gray_image.shape
        grid_h, grid_w = h // self.grid_size[0], w // self.grid_size[1]
        
        feature_vector = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # Extract each grid cell
                grid_cell = gray_image[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
                
                # Flatten grid cell and filter out zero values if they represent background
                grid_values = grid_cell.flatten()
                grid_values = grid_values[grid_values > 0]  # Ignore background pixels
                
                if len(grid_values) > 0:
                    # Calculate intensity stats and extend feature vector
                    stats = self.calculate_intensity_stats(grid_values)
                else:
                    # If the grid cell has no valid values, set default stats to 0
                    stats = [0, 0, 0, 0, 0]
                
                feature_vector.extend(stats)
        
        return feature_vector
