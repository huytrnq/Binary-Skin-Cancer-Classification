import cv2
import numpy as np
from scipy.fftpack import dct
from skimage.feature import graycomatrix, graycoprops

class ColorDescriptor:
    def __init__(self, bins, grid_x=3, grid_y=3):
        """
        Initializes the color descriptor with grid-based extraction.

        Args:
            bins (tuple): The number of bins for each channel in the HSV color space.
            grid_x (int): Number of grid divisions along the x-axis.
            grid_y (int): Number of grid divisions along the y-axis.
        """
        self.bins = bins
        self.grid_x = grid_x
        self.grid_y = grid_y

    def extract(self, image, mask=None):
        """
        Extracts color histograms from the image in the HSV color space for each grid cell.

        Args:
            image (numpy array): The image from which to extract the color features.
            mask (numpy array): The mask to apply to the image (default is None).

        Returns:
            features (list): A concatenated list of histogram values from each grid cell, normalized.
        """
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get image dimensions and calculate grid cell size
        h, w = hsv_image.shape[:2]
        cell_h, cell_w = h // self.grid_y, w // self.grid_x

        # Initialize the feature vector
        features = []

        # Loop through each grid cell
        for row in range(self.grid_y):
            for col in range(self.grid_x):
                # Define the region for the current grid cell
                x_start, x_end = col * cell_w, (col + 1) * cell_w
                y_start, y_end = row * cell_h, (row + 1) * cell_h
                
                # Extract the cell from the HSV image
                cell = hsv_image[y_start:y_end, x_start:x_end]

                # Apply mask if provided, specific to the grid cell
                cell_mask = mask[y_start:y_end, x_start:x_end] if mask is not None else None

                # Calculate the histogram for the cell
                hist = cv2.calcHist([cell], [0, 1, 2], cell_mask, self.bins, 
                                    [0, 180, 0, 256, 0, 256])

                # Normalize the histogram and flatten it
                hist = cv2.normalize(hist, hist).flatten()

                # Append the cell's histogram to the features list
                features.extend(hist)

        return features
    
    def visualize(self, image, mask=None):
        """
        Visualizes color histograms for each HSV channel and each grid cell.
        
        Args:
            image (numpy array): The input image.
            mask (numpy array): Optional mask to apply to each grid cell.
        """
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get image dimensions and calculate grid cell size
        h, w = hsv_image.shape[:2]
        cell_h, cell_w = h // self.grid_y, w // self.grid_x

        # Define the channel names for HSV
        channel_names = ["Hue", "Saturation", "Value"]

        # Loop through each grid cell and plot histograms for each channel
        fig, axs = plt.subplots(self.grid_y, self.grid_x, figsize=(self.grid_x * 4, self.grid_y * 4))
        fig.suptitle("HSV Histograms for Each Grid Cell", fontsize=16)

        for row in range(self.grid_y):
            for col in range(self.grid_x):
                # Define the region for the current grid cell
                x_start, x_end = col * cell_w, (col + 1) * cell_w
                y_start, y_end = row * cell_h, (row + 1) * cell_h

                # Extract the cell from the HSV image
                cell = hsv_image[y_start:y_end, x_start:x_end]

                # Apply mask if provided, specific to the grid cell
                cell_mask = mask[y_start:y_end, x_start:x_end] if mask is not None else None

                # Plot histograms for each channel
                for i, channel_name in enumerate(channel_names):
                    hist = cv2.calcHist([cell], [i], cell_mask, [self.bins[i]], [0, 180] if i == 0 else [0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    
                    axs[row, col].plot(hist, label=channel_name)
                    axs[row, col].set_title(f"Cell [{row+1}, {col+1}]")
                    axs[row, col].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for the main title
        plt.show()


class ColorLayoutDescriptor:
    def __init__(self, grid_x=8, grid_y=8):
        """
        Initializes the Color Layout Descriptor.

        Args:
            grid_x (int): Number of cells along the X-axis.
            grid_y (int): Number of cells along the Y-axis.
        """
        self.grid_size = (grid_x, grid_y)

    def extract(self, image, mask=None):
        """
        Extracts a Color Layout Descriptor from the image.

        Args:
            image (numpy array): The image from which to extract the color layout descriptor.
            mask (numpy array): Optional mask to apply to the image.

        Returns:
            cld_features (list): A list of DCT-transformed features for the color layout descriptor.
        """
        
        ## Apply mask to the image if provided
        if mask is not None:
            masked_image = cv2.bitwise_and(image, image, mask=mask)
        else:
            masked_image = image
        
        # Convert the image to YCrCb color space for better color separation
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # Resize the image to the specified grid size (height, width)
        resized_image = cv2.resize(ycrcb_image, self.grid_size)  # grid_size is a tuple (width, height)

        # Split the Y, Cr, and Cb channels
        y_channel, cr_channel, cb_channel = cv2.split(resized_image)

        # Apply DCT to each channel to get the compact representation
        y_dct = self.apply_dct(y_channel)
        cr_dct = self.apply_dct(cr_channel)
        cb_dct = self.apply_dct(cb_channel)

        # Combine the DCT coefficients from each channel into a single feature vector
        cld_features = np.concatenate([y_dct, cr_dct, cb_dct])

        return cld_features

    def apply_dct(self, channel):
        """
        Applies the Discrete Cosine Transform (DCT) to a color channel.

        Args:
            channel (numpy array): A single color channel.

        Returns:
            dct_coeffs (numpy array): The DCT coefficients (compact representation).
        """
        # Apply 2D DCT (Discrete Cosine Transform)
        dct_coeffs = dct(dct(channel, axis=0, norm='ortho'), axis=1, norm='ortho')

        # Extract only the top-left 3x3 coefficients (low-frequency components)
        # This is a typical way to reduce dimensionality
        return dct_coeffs[:3, :3].flatten()


class ColorCooccurrenceMatrixDescriptor:
    def __init__(self, distances=[1], angles=[0], levels=8, grid_x=1, grid_y=1):
        """
        Initializes the Color Co-occurrence Matrix Histogram.

        Args:
            distances (list): Distances for GLCM computation.
            angles (list): Angles for GLCM computation.
            levels (int): Number of quantization levels for the color channels.
            grid_x (int): Number of grids along the X-axis (width).
            grid_y (int): Number of grids along the Y-axis (height).
        """
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.grid_size = (grid_x, grid_y)

    def extract(self, image, mask=None):
        """
        Extracts the Color Co-occurrence Matrix Histogram from the image with an optional mask,
        using a grid-based approach.

        Args:
            image (numpy array): The image from which to extract the CCMH features.
            mask (numpy array): Binary mask to apply to the image (1 for ROI, 0 for background).

        Returns:
            features (list): A list of combined co-occurrence matrix histograms for each color channel
                             across all grid cells.
        """
        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Initialize the feature vector
        features = []

        # Grid size based on the specified grid shape
        h, w = hsv_image.shape[:2]
        grid_h, grid_w = h // self.grid_size[0], w // self.grid_size[1]

        # Loop over each cell in the grid
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                # Define the region for the current grid cell
                y1, y2 = row * grid_h, (row + 1) * grid_h
                x1, x2 = col * grid_w, (col + 1) * grid_w

                # Extract each channel in HSV for this grid cell
                for i in range(3):
                    channel = hsv_image[y1:y2, x1:x2, i]

                    # Apply mask if provided (focus on masked region in this grid cell)
                    if mask is not None:
                        masked_channel = np.zeros_like(channel)
                        masked_channel[mask[y1:y2, x1:x2] > 0] = channel[mask[y1:y2, x1:x2] > 0]
                    else:
                        masked_channel = channel

                    # Quantize the masked channel
                    quantized_channel = self.quantize_image(masked_channel)

                    # Compute GLCM on the quantized channel
                    glcm = graycomatrix(quantized_channel, distances=self.distances, angles=self.angles, 
                                        levels=self.levels, symmetric=True, normed=True)

                    # Calculate the GLCM histogram features for the grid cell
                    glcm_histogram = self.glcm_histogram(glcm)
                    features.extend(glcm_histogram)

        return features

    def quantize_image(self, channel):
        """
        Quantizes the image channel into the specified number of levels.

        Args:
            channel (numpy array): A single channel from the image.

        Returns:
            quantized_channel (numpy array): The quantized channel with the specified number of levels.
        """
        # Normalize the channel to the range 0-1 and multiply by the number of levels, then quantize
        quantized_channel = np.floor(channel / 256.0 * self.levels).astype(np.uint8)
        return quantized_channel

    def glcm_histogram(self, glcm):
        """
        Converts the GLCM matrix to a normalized histogram for texture analysis.

        Args:
            glcm (numpy array): GLCM matrix.

        Returns:
            hist (list): Flattened and normalized GLCM histogram based on texture properties.
        """
        # Calculate texture properties from the GLCM
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        # Create a feature vector based on these properties
        hist = [contrast, correlation, energy, homogeneity]
        return hist
