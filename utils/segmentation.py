import cv2
import numpy as np
from sklearn.cluster import KMeans

class KMeansSegmentation:
    def __init__(self, k=3, max_iter=100, random_state=42):
        """
        Initializes the KMeansSegmentation class with parameters for clustering.
        
        Args:
            k: Number of clusters for KMeans segmentation (default: 3)
            max_iter: Maximum number of iterations for the KMeans algorithm (default: 100)
            random_state: Random state for reproducibility (default: 42)
        """
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state

    def __call__(self, img):
        """
        Segments the image using KMeans clustering and displays the result.
        Args:
            img (np array): Image to segment.
        Return: 
            segmented_image: Segmented image.
        """
        # Load image
        original_shape = img.shape

        # Convert image to a 2D array of pixels (flatten)
        if len(img.shape) == 3:
            pixel_values = img.reshape((-1, 3))  # Reshape to 2D array (rows = pixels, cols = RGB)
        else:
            pixel_values = img.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)  # Convert to float32 for KMeans

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=self.k, max_iter=self.max_iter, random_state=self.random_state)
        kmeans.fit(pixel_values)
        
        # Get the cluster centers (colors) and labels (which cluster each pixel belongs to)
        centers = np.uint8(kmeans.cluster_centers_)  # Convert to uint8 for displaying
        labels = kmeans.labels_  # Each pixel's assigned cluster

        # Map each pixel to the color of its corresponding cluster center
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(original_shape)  # Reshape to the original image shape

        return segmented_image
    
    
class ThresholdingSegmentation:
    def __init__(self, method='otsu', max_value=255):
        """
        Initializes the ThresholdingSegmentation class with the specified thresholding method.

        :param method: Thresholding method to use ('otsu', 'triangle', 'binary', 'binary_inv',
                    'truncatQe', 'tozero', 'tozero_inv'). Default is 'otsu'.
        :param max_value: Maximum value to use with the thresholding. Default is 255.
        """
        self.method = method
        self.max_value = max_value
        self.threshold_type = self._get_threshold_type(method)

    def _get_threshold_type(self, method):
        """
        Maps the method string to the corresponding OpenCV thresholding flag.

        :param method: Thresholding method as a string.
        :return: OpenCV thresholding flag.
        """
        method = method.lower()
        if method == 'binary':
            return cv2.THRESH_BINARY
        elif method == 'binary_inv':
            return cv2.THRESH_BINARY_INV
        elif method == 'truncate':
            return cv2.THRESH_TRUNC
        elif method == 'tozero':
            return cv2.THRESH_TOZERO
        elif method == 'tozero_inv':
            return cv2.THRESH_TOZERO_INV
        elif method == 'otsu':
            return cv2.THRESH_BINARY + cv2.THRESH_OTSU
        elif method == 'triangle':
            return cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        else:
            raise ValueError(f"Unsupported thresholding method: {method}")

    def __call__(self, image):
        """
        Applies the thresholding segmentation to the input image.

        :param image: Input image (grayscale or color).
        :return: Tuple of (threshold value used, segmented image).
        """
        # Convert to grayscale if the image is in color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply thresholding
        if self.method in ['otsu', 'triangle']:
            # Threshold value is automatically determined
            threshold_value, segmented_image = cv2.threshold(
                gray, 0, self.max_value, self.threshold_type)
        else:
            # Manually specify a threshold value (you can adjust this as needed)
            threshold_value = 127  # Default threshold value
            threshold_value, segmented_image = cv2.threshold(
                gray, threshold_value, self.max_value, self.threshold_type)

        return segmented_image