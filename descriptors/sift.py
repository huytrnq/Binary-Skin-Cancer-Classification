import cv2
import numpy as np
from utils.utils import sliding_window

class SIFTDescriptor:
    def __init__(self, duplicate_removal=False, threshold=5):
        """
        Initializes the SIFT Descriptor.
        Args:
            duplicate_removal (bool): Whether to remove duplicate keypoints.
            threshold (float): Minimum distance between keypoints for duplicate removal.
        """
        # Create SIFT detector and descriptor
        self.sift = cv2.SIFT_create()
        self.duplicate_removal = duplicate_removal
        self.threshold = threshold
        
    def apply_mask(self, keypoints, descriptors, mask):
        """
        Applies a mask to the keypoints and descriptors to target a specific region.
        
        Args:
            keypoints (list): List of keypoints.
            descriptors (numpy array): Corresponding SIFT descriptors.
        
        Returns:
            masked_keypoints (list): List of keypoints within the mask.
            masked_descriptors (numpy array): Corresponding descriptors for keypoints within the mask.
        """
        masked_keypoints = []
        masked_descriptors = []
        
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if mask[y, x] == 255:
                masked_keypoints.append(kp)
                masked_descriptors.append(descriptors[i])
        
        # Convert list of descriptors back to numpy array
        masked_descriptors = np.array(masked_descriptors)
        
        return masked_keypoints, masked_descriptors
        
    def extract_on_windows(self, image, window_size, step_size, mask=None):
        """
        Applies SIFT on each sliding window region of the image and collects keypoints and descriptors.
        
        Args:
            image (numpy array): The input image (grayscale).
            window_size (tuple): The size of the window (height, width).
            step_size (int): The step size for sliding the window.
            mask (numpy array): Mask to mask the keypoints in ROI.
            
        Returns:
            keypoints_list (list): A list of keypoints from all windows.
            descriptors_list (list): A list of descriptors from all windows.
        """
        sift = cv2.SIFT_create()  # Initialize SIFT
        keypoints_list = []
        descriptors_list = []
        
        for (x, y, window) in sliding_window(image, window_size, step_size):
            # Apply SIFT to the window
            keypoints, descriptors = sift.detectAndCompute(window, None)
            
            if keypoints:
                # Adjust keypoint coordinates to match the original image
                for kp in keypoints:
                    kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
                keypoints_list.extend(keypoints)
                descriptors_list.append(descriptors)
        
        # Stack all descriptors into a single array
        if descriptors_list:
            descriptors_list = np.vstack(descriptors_list)
        else:
            descriptors_list = None
        
        if self.duplicate_removal:
            keypoints_list, descriptors_list = self.remove_duplicate_keypoints(keypoints_list, descriptors_list, self.threshold)
        if mask is not None:
            keypoints_list, descriptors_list = self.apply_mask(keypoints_list, descriptors_list, mask)
        
        return keypoints_list, descriptors_list
    
    def remove_duplicate_keypoints(self, keypoints, descriptors, threshold=5):
        """
        Removes duplicate keypoints that are too close to each other based on a distance threshold.
        
        Args:
            keypoints (list): List of cv2.KeyPoint objects.
            descriptors (numpy array): Corresponding SIFT descriptors.
            threshold (float): Minimum allowed distance between keypoints.
        
        Returns:
            filtered_keypoints (list): List of unique keypoints.
            filtered_descriptors (numpy array): Corresponding descriptors for unique keypoints.
        """
        if len(keypoints) == 0:
            return [], None
        
        filtered_keypoints = []
        filtered_descriptors = []
        
        # Track the indices of the keypoints we want to keep
        for i, kp1 in enumerate(keypoints):
            keep = True
            for j, kp2 in enumerate(filtered_keypoints):
                dist = np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt))
                if dist < threshold:
                    keep = False
                    break
            if keep:
                filtered_keypoints.append(kp1)
                filtered_descriptors.append(descriptors[i])
        
        # Convert list of descriptors back to numpy array
        filtered_descriptors = np.array(filtered_descriptors)
        
        return filtered_keypoints, filtered_descriptors

    def extract(self, image, mask=None):
        """
        Extracts SIFT keypoints and descriptors from the image.

        Args:
            image (numpy array): The image from which to extract the SIFT features.
            mask (numpy array): Mask to mask the keypoints in ROI.

        Returns:
            keypoints (list): List of SIFT keypoints.
            descriptors (numpy array): Array of SIFT descriptors.
        """
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect SIFT keypoints and compute the descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)
        
        if mask is not None:
            keypoints, descriptors = self.apply_mask(keypoints, descriptors, mask)

        return keypoints, descriptors

    def draw_keypoints(self, image, keypoints):
        """
        Draws keypoints on the image for visualization.

        Args:
            image (numpy array): The original image.
            keypoints (list): List of keypoints to be drawn.

        Returns:
            image_with_keypoints (numpy array): The image with keypoints drawn on it.
        """
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return image_with_keypoints
    
    
import cv2
import numpy as np
from scipy.stats import skew

class SIFTColorMomentsDescriptor:
    def __init__(self, color_space='HSV'):
        """
        Initializes the SIFT-Color Moments Descriptor.
        Args:
            color_space (str): The color space to use for computing color moments ('RGB' or 'HSV').
        """
        self.sift = cv2.SIFT_create()
        self.color_space = color_space

    def extract_sift(self, image):
        """
        Extracts SIFT keypoints and descriptors from the image.
        Args:
            image (numpy array): The input image.
        Returns:
            keypoints (list): List of keypoints.
            descriptors (numpy array): Array of SIFT descriptors.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)
        return keypoints, descriptors

    def apply_mask(self, keypoints, descriptors, mask):
        """
        Applies a mask to the keypoints and descriptors to target a specific region.
        Args:
            keypoints (list): List of keypoints.
            descriptors (numpy array): Corresponding SIFT descriptors.
            mask (numpy array): Binary mask indicating region of interest.
        Returns:
            masked_keypoints (list): List of keypoints within the mask.
            masked_descriptors (numpy array): Corresponding descriptors for keypoints within the mask.
        """
        masked_keypoints = []
        masked_descriptors = []

        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if mask[y, x] == 255:  # Keep keypoint if within the mask region
                masked_keypoints.append(kp)
                masked_descriptors.append(descriptors[i])

        # Convert list of descriptors back to numpy array
        masked_descriptors = np.array(masked_descriptors)

        return masked_keypoints, masked_descriptors

    def extract_color_moments(self, image, keypoints):
        """
        Extracts color moments (mean, standard deviation, skewness) for each keypoint region.
        Args:
            image (numpy array): The input image in BGR format.
            keypoints (list): List of keypoints from SIFT.
        Returns:
            color_moments (list): List of color moments for each keypoint.
        """
        if self.color_space == 'HSV':
            color_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            color_image = image  # Default is BGR (RGB) space

        color_moments = []

        for kp in keypoints:
            # Get keypoint coordinates and size
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size // 2)  # Half the keypoint size to create the window around the keypoint

            # Extract the region around the keypoint
            patch = color_image[max(0, y-size):min(image.shape[0], y+size), 
                                max(0, x-size):min(image.shape[1], x+size)]

            if patch.size > 0:
                # Compute color moments for each channel
                moments = []
                for channel in cv2.split(patch):
                    mean = np.mean(channel)
                    stddev = np.std(channel)
                    skewness = skew(channel.ravel())  # Flatten the channel to compute skewness
                    moments.extend([mean, stddev, skewness])
                
                color_moments.append(moments)

        return np.array(color_moments)

    def extract_sift_color_moments(self, image, mask=None):
        """
        Combines SIFT descriptors with color moments to form the SIFT-Color Moments descriptor.
        Args:
            image (numpy array): The input image.
            mask (numpy array): Optional mask to restrict the keypoints to a region of interest.
        Returns:
            final_descriptors (numpy array): Combined SIFT and color moments descriptors.
        """
        # Step 1: Extract SIFT keypoints and descriptors
        keypoints, sift_descriptors = self.extract_sift(image)

        if len(keypoints) == 0:
            return None, None  # No keypoints found, return None

        # Apply mask to restrict keypoints and descriptors to the region of interest
        if mask is not None:
            keypoints, sift_descriptors = self.apply_mask(keypoints, sift_descriptors, mask)

        if len(keypoints) == 0:
            return None, None  # No keypoints after applying the mask

        # Step 2: Extract color moments for each keypoint
        color_moments = self.extract_color_moments(image, keypoints)

        # Step 3: Concatenate SIFT descriptors and color moments
        if color_moments.size > 0:
            final_descriptors = np.hstack([sift_descriptors, color_moments])
        else:
            final_descriptors = sift_descriptors

        return keypoints, final_descriptors
