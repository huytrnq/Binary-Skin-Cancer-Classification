import cv2

class SIFTDescriptor:
    def __init__(self):
        """
        Initializes the SIFT Descriptor.
        """
        # Create SIFT detector and descriptor
        self.sift = cv2.SIFT_create()

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