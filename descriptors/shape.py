import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

class HOGDescriptor:
    def __init__(self, resize_size=(128, 128), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys'):
        """
        Initializes the HOGDescriptor class with the parameters for HOG and resizing.
        
        Args:
            resize_size (tuple): Target size to resize the images (width, height).
            orientations (int): Number of orientation bins.
            pixels_per_cell (tuple): Size of each cell in pixels.
            cells_per_block (tuple): Number of cells in each block.
            block_norm (str): Block normalization method.
        """
        self.resize_size = resize_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def resize_image(self, image):
        """
        Resizes the input image to the target size.
        
        Args:
            image (numpy array): Input image.
            
        Returns:
            resized_image (numpy array): Resized image.
        """
        resized_image = cv2.resize(image, self.resize_size, interpolation=cv2.INTER_AREA)
        return resized_image

    def extract(self, image):
        """
        Extracts HOG descriptors from the input image after resizing.
        
        Args:
            image (numpy array): Input image.
            
        Returns:
            hog_features (numpy array): HOG descriptors (feature vector), flattened to ensure consistency.
            hog_image_rescaled (numpy array): HOG image for visualization.
        """
        # Resize the image to the target size
        resized_image = self.resize_image(image)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
        # Compute HOG descriptors and visualization
        hog_features, hog_image = hog(gray_image, orientations=self.orientations,
                                    pixels_per_cell=self.pixels_per_cell,
                                    cells_per_block=self.cells_per_block,
                                    block_norm=self.block_norm, visualize=True)

        # Flatten HOG features to ensure it's a 1D array
        hog_features = hog_features.flatten()

        # Rescale the HOG image for better contrast visualization
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        
        return hog_features

    def visualize(self, image, hog_image):
        """
        Visualizes the HOG descriptors and the original image.
        
        Args:
            image (numpy array): Original input image.
            hog_image (numpy array): HOG descriptor visualization image.
        """
        # Display the original image and the HOG image side by side
        plt.figure(figsize=(10, 5))

        # Show original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        # Show HOG image
        plt.subplot(1, 2, 2)
        plt.imshow(hog_image, cmap='gray')
        plt.title("HOG Descriptor Visualization")
        plt.axis("off")

        plt.show()