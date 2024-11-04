## Augmentation
import os
import cv2
import random
import numpy as np

class ImageAugmentor:
    def __init__(self, input_folder, output_folder, augment=5):
        """
        ImageAugmentor for generating augmented images with random flips, rotation, and brightness adjustment.

        Args:
            input_folder (str): Path to the folder containing original images.
            output_folder (str): Path to the folder where augmented images will be saved.
            augment (int): Number of augmented images to generate per input image.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.augment = augment
        os.makedirs(self.output_folder, exist_ok=True)
    
    def random_flip(self, img):
        """Randomly flip the image horizontally or vertically."""
        flip_type = random.choice([-1, 0, 1])  # -1: both, 0: vertical, 1: horizontal
        return cv2.flip(img, flip_type)

    def random_rotation(self, img):
        """Randomly rotate the image by 0, 90, 180, or 270 degrees."""
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            return img
        else:
            img_center = tuple(np.array(img.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
            return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    def random_brightness(self, img):
        """Randomly adjust the brightness of the image."""
        value = random.randint(-50, 50)  # Brightness change value
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)
        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def augment_image(self, img):
        """Apply random flip, rotation, and brightness adjustments to the image."""
        img = self.random_flip(img)
        img = self.random_rotation(img)
        img = self.random_brightness(img)
        return img

    def generate_augmented_images(self):
        """Generate augmented images for each image in the input folder."""
        for filename in os.listdir(self.input_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.input_folder, filename)
                img = cv2.imread(img_path)
                
                for i in range(self.augment):
                    augmented_img = self.augment_image(img)
                    augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
                    augmented_path = os.path.join(self.output_folder, augmented_filename)
                    cv2.imwrite(augmented_path, augmented_img)
                    print(f"Generated augmented image: {augmented_path}")


if __name__ == "__main__":
    augmentor = ImageAugmentor(input_folder='../Data/Challenge2/train/scc/', output_folder='../Data/Challenge2/train/scc/', augment=7)
    augmentor.generate_augmented_images()