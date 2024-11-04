import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops


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
                
                lbp_masked = lbp[grid > 0]
                
                # Calculate LBP histogram for the current grid
                lbp_hist, _ = np.histogram(lbp_masked.ravel(), bins=np.arange(0, self.n_points + 3), range=(0, self.n_points + 2))

                # Append the histogram to the concatenated histograms
                concatenated_hist.extend(lbp_hist)

                if self.visualize:
                    # Store the LBP result into the lbp_image
                    lbp_image[start_y:end_y, start_x:end_x] = lbp

        # Convert to numpy array
        return np.array(concatenated_hist), lbp_image

class GLCMDescriptor:
    def __init__(self, distances=[1], angles=[0], levels=8, grid_x=1, grid_y=1, visualize=False):
        """
        Initializes the GLCMDescriptor with grid options.

        Args:
            distances (list): Distances for GLCM computation.
            angles (list): Angles for GLCM computation.
            levels (int): Number of quantization levels for the GLCM.
            grid_x (int): Number of grids along the X-axis (width).
            grid_y (int): Number of grids along the Y-axis (height).
            visualize (bool): Whether to visualize the quantized image and GLCM matrix for the whole image.
        """
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.visualize = visualize

    def extract(self, image, mask=None):
        """
        Extracts GLCM features from the image, dividing it into grids.

        Args:
            image (numpy array): The image from which to extract GLCM features.
            mask (numpy array): Optional mask to apply to the image.

        Returns:
            concatenated_features (list): Concatenated GLCM features from all grids.
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

        # Initialize the concatenated features list
        concatenated_features = []

        if self.visualize:
            # Create an empty image to hold the entire quantized result
            quantized_image_full = np.zeros_like(masked_image)
        else:
            quantized_image_full = None

        # Loop over the grids
        for i in range(self.grid_y):
            for j in range(self.grid_x):
                # Extract the sub-region (grid) from the image
                start_y = i * grid_height
                end_y = start_y + grid_height
                start_x = j * grid_width
                end_x = start_x + grid_width

                grid = masked_image[start_y:end_y, start_x:end_x]

                # Quantize the grid to a fixed number of gray levels (for GLCM)
                quantized_grid = self.quantize_image(grid)

                if self.visualize:
                    # Place the quantized grid back into the full image
                    quantized_image_full[start_y:end_y, start_x:end_x] = quantized_grid

                # Compute GLCM for the current grid
                glcm = graycomatrix(quantized_grid, distances=self.distances, angles=self.angles, levels=self.levels,
                                    symmetric=True, normed=True)

                # Extract GLCM features for the grid
                contrast = graycoprops(glcm, 'contrast').mean()
                correlation = graycoprops(glcm, 'correlation').mean()
                energy = graycoprops(glcm, 'energy').mean()
                homogeneity = graycoprops(glcm, 'homogeneity').mean()

                # Combine GLCM features into a list
                glcm_features = [contrast, correlation, energy, homogeneity]

                # Append the features to the concatenated features list
                concatenated_features.extend(glcm_features)

        return concatenated_features, quantized_image_full

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
    
    
import cv2
import numpy as np

class GaborFilterDescriptor:
    """
    Extracts texture features using Gabor filters at multiple orientations and frequencies.
    """
    
    def __init__(self, frequencies=[0.1, 0.2, 0.3], orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """
        Initializes Gabor filter parameters.

        Args:
            frequencies (list): List of frequencies for the Gabor filters.
            orientations (list): List of orientations (angles in radians) for the Gabor filters.
        """
        self.frequencies = frequencies
        self.orientations = orientations

    def extract(self, image, mask=None):
        """
        Applies Gabor filters to the image at different orientations and frequencies, 
        and returns the filtered responses as texture features.

        Args:
            image (numpy array): The grayscale image for feature extraction.
            mask (numpy array): Optional binary mask to specify the region of interest.

        Returns:
            features (list): Concatenated list of mean and standard deviation of filtered responses.
        """
        # Ensure the image is grayscale
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Initialize the feature list
        features = []

        # Apply Gabor filters across different orientations and frequencies
        for frequency in self.frequencies:
            for theta in self.orientations:
                # Create Gabor kernel
                kernel = self._create_gabor_kernel(frequency, theta)
                
                # Apply filter to the image
                filtered_image = cv2.filter2D(gray_image, cv2.CV_32F, kernel)
                
                # If mask is provided, apply mask
                if mask is not None:
                    filtered_image = cv2.bitwise_and(filtered_image, filtered_image, mask=mask)

                # Calculate mean and standard deviation of the filtered response
                mean_val = np.mean(filtered_image)
                std_val = np.std(filtered_image)
                
                # Append the feature values
                features.extend([mean_val, std_val])

        return features

    def _create_gabor_kernel(self, frequency, theta, sigma=4.0, gamma=0.5, psi=0):
        """
        Creates a Gabor kernel with the specified parameters.

        Args:
            frequency (float): Frequency of the sinusoidal function.
            theta (float): Orientation of the Gabor filter in radians.
            sigma (float): Standard deviation of the Gaussian envelope.
            gamma (float): Spatial aspect ratio.
            psi (float): Phase offset.

        Returns:
            kernel (numpy array): The Gabor kernel.
        """
        # Size of the kernel; calculated to include the entire Gaussian envelope
        kernel_size = int(1.5 * sigma / frequency)
        if kernel_size % 2 == 0: kernel_size += 1  # Ensure kernel size is odd

        # Generate the Gabor kernel
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, 1.0/frequency, gamma, psi, ktype=cv2.CV_32F)
        return kernel
