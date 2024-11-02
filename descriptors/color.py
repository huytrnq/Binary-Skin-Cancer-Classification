import cv2
import numpy as np
from utils.utils import weighted_mask, distance_transform_weighting
from skimage.feature import graycomatrix, graycoprops


class ColorDescriptor:
    def __init__(self, bins):
        """
        Initializes the color descriptor.
        
        Args:
            bins (tuple): The number of bins for each channel in the HSV color space.
        """
        self.bins = bins

    def extract(self, image, mask=None):
        """
        Extracts a color histogram from an image in the HSV color space.

        Args:
            image (numpy array): The image from which to extract the color features.
            mask (numpy array): The mask to apply to the image (default is None).

        Returns:
            features (list): A flattened list of the histogram values normalized.
        """
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Initialize the color histogram
        features = []
        
        mask = distance_transform_weighting(image, mask) if mask is not None else None

        # Extract the color histogram from the entire image
        hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, self.bins, 
                            [0, 180, 0, 256, 0, 256])

        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()

        # Add the histogram to the feature vector
        features.extend(hist)

        return features
    

class MultiColorSpaceDescriptor:
    def __init__(self, bins=(8, 8, 8)):
        """
        Initializes the multi-color space descriptor.

        Args:
            bins (tuple): The number of bins for each channel in each color space.
        """
        self.bins = bins

    def extract(self, image, mask=None):
        """
        Extracts color histograms from the image in multiple color spaces and concatenates them.

        Args:
            image (numpy array): The image from which to extract the color features.
            mask (numpy array): The mask to apply to the image (default is None).

        Returns:
            features (list): A concatenated list of normalized histogram values from all color spaces.
        """
        # Initialize the feature vector
        features = []

        # Extract features from each color space and concatenate
        features.extend(self.extract_hsv(image, mask))
        # features.extend(self.extract_rgb(image, mask))
        features.extend(self.extract_lab(image, mask))

        return features

    def extract_hsv(self, image, mask):
        """
        Extracts a color histogram in the HSV color space.

        Args:
            image (numpy array): The image from which to extract the color features.
            mask (numpy array): The mask to apply to the image (default is None).

        Returns:
            features (list): A flattened list of the histogram values normalized.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.tolist()

    def extract_rgb(self, image, mask):
        """
        Extracts a color histogram in the RGB color space.

        Args:
            image (numpy array): The image from which to extract the color features.
            mask (numpy array): The mask to apply to the image (default is None).

        Returns:
            features (list): A flattened list of the histogram values normalized.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([rgb_image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.tolist()

    def extract_ycbcr(self, image, mask):
        """
        Extracts a color histogram in the YCbCr color space.

        Args:
            image (numpy array): The image from which to extract the color features.
            mask (numpy array): The mask to apply to the image (default is None).

        Returns:
            features (list): A flattened list of the histogram values normalized.
        """
        ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        hist = cv2.calcHist([ycbcr_image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.tolist()

    def extract_lab(self, image, mask):
        """
        Extracts a color histogram in the Lab color space.

        Args:
            image (numpy array): The image from which to extract the color features.
            mask (numpy array): The mask to apply to the image (default is None).

        Returns:
            features (list): A flattened list of the histogram values normalized.
        """
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        hist = cv2.calcHist([lab_image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.tolist()



class ColorLayoutDescriptor:
    def __init__(self, grid_x=8, grid_y=8, dct_size=8):
        """
        Initializes the Color Layout Descriptor with grid-based extraction.

        Args:
            grid_x (int): Number of grid cells along the X-axis.
            grid_y (int): Number of grid cells along the Y-axis.
            dct_size (int): Number of DCT coefficients to keep (from top-left corner).
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.dct_size = dct_size

    def extract(self, image, mask=None):
        """
        Extracts a Color Layout Descriptor from the image using grid-based extraction.

        Args:
            image (numpy array): The image from which to extract the color layout descriptor.
            mask (numpy array): Optional mask to apply to the image.

        Returns:
            cld_features (list): A concatenated list of DCT-transformed features for each grid cell.
        """
        
        # Apply mask to the image if provided
        if mask is not None:
            # masked_image = cv2.bitwise_and(image, image, mask=mask)
            mask = distance_transform_weighting(image, mask) if mask is not None else None
        else:
            masked_image = image

        # Convert the image to YCrCb color space for better color separation
        ycrcb_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2YCrCb)

        # Get image dimensions and calculate grid cell size
        h, w = ycrcb_image.shape[:2]
        cell_h, cell_w = h // self.grid_y, w // self.grid_x

        # Initialize the feature vector
        cld_features = []

        # Loop through each grid cell
        for row in range(self.grid_y):
            for col in range(self.grid_x):
                # Define the region for the current grid cell
                x_start, x_end = col * cell_w, (col + 1) * cell_w
                y_start, y_end = row * cell_h, (row + 1) * cell_h

                # Extract the cell from the YCrCb image
                cell = ycrcb_image[y_start:y_end, x_start:x_end]

                # Resize cell to an even dimension if necessary
                if cell.shape[0] % 2 != 0 or cell.shape[1] % 2 != 0:
                    cell = cv2.resize(cell, (cell.shape[1] + cell.shape[1] % 2, cell.shape[0] + cell.shape[0] % 2))

                # Split the Y, Cr, and Cb channels
                y_channel, cr_channel, cb_channel = cv2.split(cell)

                # Apply DCT to each channel and keep top-left `dct_size` coefficients
                y_dct = self.apply_dct(y_channel)
                cr_dct = self.apply_dct(cr_channel)
                cb_dct = self.apply_dct(cb_channel)

                # Concatenate the DCT coefficients for this grid cell
                cell_features = np.concatenate([y_dct, cr_dct, cb_dct])

                # Add the grid cell's features to the overall feature vector
                cld_features.extend(cell_features)

        return cld_features

    def apply_dct(self, channel):
        """
        Applies Discrete Cosine Transform (DCT) to a single channel and keeps the top-left
        `dct_size` coefficients.

        Args:
            channel (numpy array): A single color channel from the image.

        Returns:
            dct_coefficients (numpy array): The top-left DCT coefficients as a flattened array.
        """
        # Apply DCT
        dct = cv2.dct(np.float32(channel))
        
        # Keep only the top-left dct_size x dct_size block
        dct_block = dct[:self.dct_size, :self.dct_size]

        # Flatten the block to create a feature vector
        dct_coefficients = dct_block.flatten()

        return dct_coefficients



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
