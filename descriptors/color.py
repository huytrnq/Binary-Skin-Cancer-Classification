import cv2
import numpy as np
from scipy.fftpack import dct
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

        # Extract the color histogram from the entire image
        hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, self.bins, 
                            [0, 180, 0, 256, 0, 256])

        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()

        # Add the histogram to the feature vector
        features.extend(hist)

        return features


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
    def __init__(self, distances=[1], angles=[0], levels=8):
        """
        Initializes the Color Co-occurrence Matrix Histogram.

        Args:
            distances (list): Distances for GLCM computation.
            angles (list): Angles for GLCM computation.
            levels (int): Number of quantization levels for the color channels.
        """
        self.distances = distances
        self.angles = angles
        self.levels = levels

    def extract(self, image, mask=None):
        """
        Extracts the Color Co-occurrence Matrix Histogram from the image with an optional mask.

        Args:
            image (numpy array): The image from which to extract the CCMH features.
            mask (numpy array): Binary mask to apply to the image (1 for ROI, 0 for background).

        Returns:
            features (list): A list of combined co-occurrence matrix histograms for each color channel.
        """
        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Initialize the feature vector
        features = []

        # Apply mask to each channel in HSV (if a mask is provided)
        for i in range(3):
            channel = hsv_image[:, :, i]

            # If mask is provided, set pixels outside mask to 0
            if mask is not None:
                masked_channel = np.zeros_like(channel)
                masked_channel[mask > 0] = channel[mask > 0]  # Use only pixels inside the mask
            else:
                masked_channel = channel

            # Quantize the masked channel
            quantized_channel = self.quantize_image(masked_channel)

            # Compute GLCM on the quantized channel
            glcm = graycomatrix(quantized_channel, distances=self.distances, angles=self.angles, levels=self.levels,
                                symmetric=True, normed=True)

            # Calculate the GLCM histogram features
            glcm_histogram = self.glcm_histogram(glcm)
            features.extend(glcm_histogram)

        return features

    def quantize_image(self, channel):
        """
        Quantizes the image channel into the specified number of levels.

        Args:
            channel (numpy array): A single channel from the image.

        Returns:
            quantized_channel (numpy array): The quantized channel with the number of levels.
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
            hist (list): Flattened and normalized GLCM histogram.
        """
        # Calculate texture properties from the GLCM
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        # Create a feature vector based on these properties
        hist = [contrast, correlation, energy, homogeneity]
        return hist