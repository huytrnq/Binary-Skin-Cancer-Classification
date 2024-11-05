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
    
    
class CannyEdgeDetector:
    def __init__(self, low_threshold=50, high_threshold=150, aperture_size=3, L2gradient=False):
        """
        Initializes the CannyEdgeDetector with specified parameters.

        :param low_threshold: First threshold for the hysteresis procedure.
        :param high_threshold: Second threshold for the hysteresis procedure.
        :param aperture_size: Aperture size for the Sobel operator (must be odd and between 3 and 7).
        :param L2gradient: Boolean indicating whether to use a more accurate L2 norm.
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.aperture_size = aperture_size
        self.L2gradient = L2gradient

    def __call__(self, image):
        """
        Detects edges in the input image using the Canny algorithm.

        :param image: Input image (grayscale or color).
        :return: Edge-detected image.
        """
        # Convert to grayscale if the image is in color
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Apply Gaussian Blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)

        # Perform Canny edge detection
        edges = cv2.Canny(
            blurred_image,
            self.low_threshold,
            self.high_threshold,
            apertureSize=self.aperture_size,
            L2gradient=self.L2gradient
        )

        return edges
    
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
from tqdm import tqdm
from sklearn.utils import resample
from utils.transforms import HairRemoval
from utils.dataloader import DataLoader
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu

# Utility Functions

def fill_median(image, mask):
    mask = mask // 255
    filled_image = image.copy()
    for i in range(3):
        channel = image[:, :, i]
        valid_pixels = channel[mask != 0]
        median_value = np.median(valid_pixels)
        filled_image[:, :, i][mask == 0] = median_value
    return filled_image

def apply_otsu(image, mask=None):
    if image.dtype != np.uint8:
        image = np.uint8(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mask is None:
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = np.uint8(mask)
        masked_pixels = gray[mask != 0]
        if len(masked_pixels) > 0:
            otsu_thresh_value = threshold_otsu(masked_pixels)
            binary_mask = np.zeros_like(gray)
            binary_mask[mask != 0] = (gray[mask != 0] >= otsu_thresh_value) * 255
        else:
            binary_mask = np.zeros_like(gray)
    masked_image = cv2.bitwise_and(image, image, mask=binary_mask.astype(np.uint8))
    return binary_mask, masked_image

def apply_clahe(image, mask=None, clip_limit=3.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    if mask is not None:
        mask = (mask > 0).astype(np.uint8)
        l[mask != 0] = l_clahe[mask != 0]
    else:
        l = l_clahe
    lab_clahe = cv2.merge((l, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def apply_mean_shift_without_masked_area(image, mask, sp=30, sr=30, max_level=1):
    mask = (mask > 0).astype(np.uint8)
    modified_image = image.copy()
    modified_image[mask == 0] = [0, 0, 0]
    mean_shifted_image = cv2.pyrMeanShiftFiltering(modified_image, sp, sr, maxLevel=max_level)
    mean_shifted_image[mask == 0] = image[mask == 0]
    return mean_shifted_image

def apply_em_segmentation(image, mask=None, n_components=2):
    img_float = np.float32(image.reshape(-1, 3))
    if mask is not None:
        mask = (mask > 0).astype(np.uint8)
        mask_flat = mask.flatten()
        valid_pixels = img_float[mask_flat != 0]
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(valid_pixels)
        labels = gmm.predict(valid_pixels)
        clustered_pixels = gmm.means_[labels].astype(np.uint8)
        segmented_image = image.copy().reshape(-1, 3)
        segmented_image[mask_flat != 0] = clustered_pixels
        segmented_image = segmented_image.reshape(image.shape)
    else:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(img_float)
        labels = gmm.predict(img_float)
        clustered_pixels = gmm.means_[labels].astype(np.uint8)
        segmented_image = clustered_pixels.reshape(image.shape)
    return segmented_image

def create_corner_rectangles_mask(image_shape, rect_size):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    rect_h, rect_w = rect_size, rect_size
    cv2.rectangle(mask, (0, 0), (rect_w, rect_h), 255, -1)
    cv2.rectangle(mask, (width - rect_w, 0), (width, rect_h), 255, -1)
    cv2.rectangle(mask, (0, height - rect_h), (rect_w, height), 255, -1)
    cv2.rectangle(mask, (width - rect_w, height - rect_h), (width, height), 255, -1)
    return mask

def create_black_background_mask(gray):
    _, background_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    h, w = gray.shape
    center = (w // 2, h // 2)
    radius = min(h, w) // 3
    exclusion_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(exclusion_mask, center, radius, 255, -1)
    exclusion_mask = cv2.bitwise_not(exclusion_mask)
    final_mask = cv2.bitwise_and(background_mask, exclusion_mask)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.bitwise_not(final_mask)

def calculate_center_distance_outside_circle(binary_image, radius_threshold=5):
    h, w = binary_image.shape
    center = (w // 2, h // 2)
    diagonal = np.sqrt(center[0]**2 + center[1]**2)
    radius = diagonal / 2
    exclusion_mask = np.zeros_like(binary_image, dtype=np.uint8)
    cv2.circle(exclusion_mask, center, int(radius), 255, -1)
    masked_image = cv2.bitwise_or(binary_image, exclusion_mask)
    distance_transform = cv2.distanceTransform(masked_image, cv2.DIST_L2, 5)
    center_distance = distance_transform[center[1], center[0]]
    new_radius = center_distance - radius_threshold

    new_mask = np.zeros_like(binary_image, dtype=np.uint8)
    cv2.circle(new_mask, center, int(new_radius), 255, -1)
    return new_mask

def exclusion(binary_image, radius_threshold=5):
    h, w = binary_image.shape
    center = (w // 2, h // 2)
    diagonal = np.sqrt(center[0]**2 + center[1]**2)
    radius = diagonal / 2
    exclusion_mask = np.zeros_like(binary_image, dtype=np.uint8)
    cv2.circle(exclusion_mask, center, int(radius), 255, -1)
    return exclusion_mask



def check_histogram_count(im_gray, mask):
    """
    Checks the histogram count in the masked area for grayscale values between 0 and 20.

    Parameters:
    - im_gray: Grayscale image
    - mask: Binary mask to limit the region to check the histogram

    Returns:
    - count: Sum of pixel values in the range 0-20 in the histogram
    """
    im_hist = cv2.calcHist([im_gray], [0], mask, [256], [0, 256])
    count = sum(im_hist[0:20])
    return count

def segmentation_pipeline_1(img, im_gray, thresholded):
    """
    Segmentation pipeline when the histogram count is greater than or equal to 1500.
    This pipeline involves CLAHE, Gaussian blur, and Mean Shift Segmentation.

    Parameters:
    - img: Input image in RGB format
    - im_gray: Grayscale image
    - thresholded: Masked binary image

    Returns:
    - binary_seg: Final binary segmented image
    """
    thresholded = cv2.erode(thresholded, cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(80, 80)))
    cla = apply_clahe(img, thresholded)
    cla = cv2.GaussianBlur(cla, (15, 15), 0)
    seg = apply_mean_shift_without_masked_area(cla, thresholded, sp=50, sr=50, max_level=1)
    seg_1 = apply_em_segmentation(seg, thresholded)
    seg_1 = fill_median(seg_1, thresholded)
    seg_gr = cv2.cvtColor(seg_1, cv2.COLOR_RGB2GRAY)
    _, binary_seg = cv2.threshold(seg_gr, 127, 255, cv2.THRESH_OTSU)
    binary_seg[np.where(thresholded == 0)] = 255
    return binary_seg

def segmentation_pipeline_2(img):
    """
    Segmentation pipeline when the histogram count is below 1500.
    This pipeline involves CLAHE, Gaussian blur, and Mean Shift Segmentation without masking.

    Parameters:
    - img: Input image in RGB format

    Returns:
    - binary_seg: Final binary segmented image
    """
    cla = apply_clahe(img)
    cla = cv2.GaussianBlur(cla, (15, 15), 0)
    seg = cv2.pyrMeanShiftFiltering(cla, 30, 30, maxLevel=1)
    seg_1 = apply_em_segmentation(seg)
    seg_gr = cv2.cvtColor(seg_1, cv2.COLOR_RGB2GRAY)
    _, binary_seg = cv2.threshold(seg_gr, 127, 255, cv2.THRESH_OTSU)
    # binary_seg_circ = binary_seg.copy()
    # thresholded = calculate_center_distance_outside_circle(binary_seg)
    # binary_seg_circ[np.where(thresholded == 0)] = 255
    return binary_seg

def geodesic_dilation(masked_image, marker_image, erosion_kernel_size=(15, 15), max_iterations=100):
    """
    Applies erosion to reduce small objects in the masked image, followed by geodesic dilation
    using the marker image to restore the main structure.

    Parameters:
    - masked_image (numpy.ndarray): The binary masked image.
    - marker_image (numpy.ndarray): The marker image used for geodesic dilation.
    - erosion_kernel_size (tuple): Size of the structuring element for erosion.
    - max_iterations (int): Maximum number of iterations for geodesic dilation.

    Returns:
    - geodesic_dilated (numpy.ndarray): Result after geodesic dilation.
    """
    # Step 1: Erode the masked image with a large structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erosion_kernel_size)
    eroded_mask = cv2.erode(masked_image, kernel)
    plt.figure()
    plt.imshow(eroded_mask, cmap='gray')
    plt.show()
    # Step 2: Initialize geodesic dilation
    prev_marker = eroded_mask.copy()
    while True:
        # Dilate the previous marker
        dilated_marker = cv2.dilate(prev_marker, kernel)
        
        # Perform the geodesic dilation: intersection between dilated marker and eroded mask
        geodesic_dilated = cv2.bitwise_and(dilated_marker, marker_image)
        
        # Check if convergence is reached
        if np.array_equal(geodesic_dilated, prev_marker):
            break
            
        # Update previous marker
        prev_marker = geodesic_dilated.copy()
        
        # Limit the iterations if specified
        if max_iterations and max_iterations <= 0:
            break
        max_iterations -= 1

    return geodesic_dilated, eroded_mask


def apply_circular_mask(binary_mask, th = 5):
    # Get the dimensions of the image
    h, w = binary_mask.shape
    center = (w // 2, h // 2)
    
    # Calculate the radius as half the distance from center to a corner
    radius = int(np.sqrt(center[0]**2 + center[1]**2) / 2)
    
    # Create a circular mask
    circular_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    cv2.circle(circular_mask, center, radius - th, 255, -1)
    
    # Perform a bitwise AND to keep only contours inside the circular mask
    masked_result = cv2.bitwise_and(binary_mask, circular_mask)
    
    return masked_result

def count_white_pixels_in_circle(binary_image):
    # Get image dimensions
    h, w = binary_image.shape
    center = (w // 2, h // 2)
    
    # Calculate the distance from center to corner
    corner_distance = np.sqrt((w // 2) ** 2 + (h // 2) ** 2)
    radius = int(corner_distance / 4)
    
    # Create a circular mask with the calculated radius
    circular_mask = np.zeros_like(binary_image, dtype=np.uint8)
    cv2.circle(circular_mask, center, radius, 255, -1)
    
    # Count the white pixels inside the circular region in the binary image
    white_pixels_inside_circle = cv2.countNonZero(cv2.bitwise_and(binary_image, circular_mask))
    
    # Calculate the percentage of white pixels inside the circle
    total_circle_pixels = cv2.countNonZero(circular_mask)
    white_pixel_percentage = (white_pixels_inside_circle / total_circle_pixels) * 100 if total_circle_pixels > 0 else 0
    
    return white_pixels_inside_circle, white_pixel_percentage

def main(data_folder, save_path, mode='train', save=False):
    type_count1, type_count2 = 0, 0
    dataloader = DataLoader(data_folder, mode)
    
    for i, (img, label, path) in tqdm(enumerate(dataloader), total=len(dataloader)):
        name = os.path.basename(path)
        
        img_org = img.copy()
        
        # Initial hair removal
        result, hair = HairRemoval(kernel_size=(15, 15), inpaint_radius=5)(img)
        if hair > 15:
            result, hair = HairRemoval(kernel_size=(30, 30), inpaint_radius=5)(img)
        
        img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Create mask for corners and check histogram
        msk = create_corner_rectangles_mask(np.shape(img), 250)
        count = check_histogram_count(im_gray, msk)
        
        # Apply segmentation pipeline based on histogram count
        if count >= 1500:
            _, binary = cv2.threshold(im_gray, 20, 255, cv2.THRESH_BINARY)
            thresholded = calculate_center_distance_outside_circle(binary)
            binary_seg = segmentation_pipeline_1(img, im_gray, thresholded)
        else:
            binary_seg = segmentation_pipeline_2(img)
        
        binary_seg = cv2.bitwise_not(binary_seg)
        
        # Use segmentation result as mask and proceed with further checks
        mask_thresh = count_white_pixels_in_circle(binary_seg)[1]
        
        if mask_thresh < 20:
            im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, binary_otsu = cv2.threshold(im_gray, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            msk = create_corner_rectangles_mask(np.shape(img), 250)
            count = check_histogram_count(im_gray, msk)
            
            if count >= 1500:
                _, binary = cv2.threshold(im_gray, 20, 255, cv2.THRESH_BINARY)
                thresholded = calculate_center_distance_outside_circle(binary)
                thresholded = cv2.erode(thresholded, cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(80, 80)))
                cla = apply_clahe(result, thresholded)
                cla = cv2.GaussianBlur(cla, (15, 15), 0)
                otsu, _ = apply_otsu(cla, thresholded)
            else:
                cla = apply_clahe(result)
                cla = cv2.GaussianBlur(cla, (15, 15), 0)
                otsu, _ = apply_otsu(cla)
            
            mask = cv2.bitwise_not(otsu)
        else:
            mask = binary_seg
        
        # Further morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.erode(mask, kernel)
        
        # Circular mask and geodesic dilation
        circ_masked = apply_circular_mask(mask)
        geo, eroded_mask = geodesic_dilation(circ_masked, mask, erosion_kernel_size=(50, 50))
        
        # Set final mask based on geodesic dilation
        final_mask = geo if np.any(geo) else mask
        final_mask = cv2.dilate(final_mask, kernel)
        
        # Save final mask if required
        if save:
            cv2.imwrite(os.path.join(save_path, name), final_mask)
        
        # Stop processing if we reach the sample limit
        if type_count1 >= 8000 and type_count2 >= 8000:
            break

if __name__ == "__main__":
    # Work folder paths
    work_folder = "/Users/sumeetdash/MAIA/Semester_3/CAD"
    data_folder = os.path.join(work_folder, 'Data')
    save_path = os.path.join(work_folder, 'Mask_val')
    mode = 'train'  # Change as needed
    
    # Call main function
    main(data_folder, save_path, mode=mode, save=True)  # Set save=True to enable mask saving

