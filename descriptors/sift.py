import cv2
import numpy as np
from utils.utils import sliding_window

class SIFTDescriptor:
    def __init__(self, duplicate_removal=False, threshold=5):
        """
        Initializes the SIFT Descriptor.
        """
        # Create SIFT detector and descriptor
        self.sift = cv2.SIFT_create()
        self.duplicate_removal = duplicate_removal
        self.threshold = threshold
        
    def extract_on_windows(self, image, window_size, step_size):
        """
        Applies SIFT on each sliding window region of the image and collects keypoints and descriptors.
        
        Args:
            image (numpy array): The input image (grayscale).
            window_size (tuple): The size of the window (height, width).
            step_size (int): The step size for sliding the window.
            
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

    def extract(self, image):
        """
        Extracts SIFT keypoints and descriptors from the image.

        Args:
            image (numpy array): The image from which to extract the SIFT features.

        Returns:
            keypoints (list): List of SIFT keypoints.
            descriptors (numpy array): Array of SIFT descriptors.
        """
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect SIFT keypoints and compute the descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)

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