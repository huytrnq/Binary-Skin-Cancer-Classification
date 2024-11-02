import cv2
import numpy as np

class Composer:
    def __init__(self, transforms):
        """
        Initialize Composer with transforms.

        Args:
            transforms (list): List of transforms to apply.
        """
        self.transforms = transforms

    def __call__(self, img):
        """
        Apply transforms to the image.

        Args:
            img (numpy.ndarray): Image to transform.

        Returns:
            numpy.ndarray: Transformed image.
        """
        for transform in self.transforms:
            img = transform(img)
        return img
    

class CentricCropping:
    def __init__(self, grid_size):
        """
        Initialize ObjectCentricCropping with crop size.

        Args:
            grid_size (tuple): Divide the image into grid_size then take only the center
        """
        self.grid_size = grid_size

    def __call__(self, img):
        """
        Crop the image around the object.

        Args:
            img (numpy.ndarray): Image to crop.

        Returns:
            numpy.ndarray: Cropped image.
        """
        # Convert image to grayscale
        h, w = img.shape[:2]
        
        # Calculate the center of the object
        center_x = w // 2
        center_y = h // 2
        
        # Calculate the crop size
        crop_size_x = w // self.grid_size[0]
        crop_size_y = h // self.grid_size[1]
        
        # Calculate the crop coordinates
        crop_x = center_x - crop_size_x // 2
        crop_y = center_y - crop_size_y // 2
        
        # Crop the image
        cropped = img[crop_y:crop_y + crop_size_y, crop_x:crop_x + crop_size_x]
        return cropped
    
class HairRemoval:
    def __init__(self, kernel_size=(15, 15), inpaint_radius=1):
        """Hair Removal Using Morphological Operations and Inpainting

        Args:
            kernel_size (tuple, optional): _description_. Defaults to (15, 15).
            inpaint_radius (int, optional): _description_. Defaults to 1.
        """
        self.kernel_size = kernel_size
        self.inpaint_radius = inpaint_radius
    
    def __call__(self, img):
        """
        Remove hair from the image.

        Args:
            img (numpy.ndarray): Image to remove hair from.

        Returns:
            numpy.ndarray: Image with hair removed.
        """
        # Check if the image is gray
        if len(img.shape) == 3:
            if img.shape[2] == 3:    
                im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            im_gray = img
        # Convert image to grayscale
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)  # You can adjust kernel size based on hair thickness
        # Apply a blackhat filter to highlight the hair
        blackhat = cv2.morphologyEx(im_gray, cv2.MORPH_BLACKHAT, kernel)

        # Apply a binary threshold to extract the hair regions
        _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        # Perform morphological closing to clean up the mask
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Adjust size if necessary
        closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel_close)

        # Inpaint the original image using the hair mask to remove the hair
        result = cv2.inpaint(img, closing, inpaintRadius=self.inpaint_radius, flags=cv2.INPAINT_TELEA)
        return result
    

class GaussianBlur:
    def __init__(self, kernel_size=(15, 15)):
        """
        Initialize Blur with kernel size.

        Args:
            kernel_size (tuple): Size of the kernel.
        """
        self.kernel_size = kernel_size

    def __call__(self, img):
        """
        Apply blur to the image.

        Args:
            img (numpy.ndarray): Image to blur.

        Returns:
            numpy.ndarray: Blurred image.
        """
        return cv2.GaussianBlur(img, self.kernel_size, 0)
    
    