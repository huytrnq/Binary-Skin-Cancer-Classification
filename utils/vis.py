import cv2
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import cv2
import numpy as np

class MatplotlibVisualizer:
    def __init__(self):
        pass

    def show_image(self, img, title, color_map='gray'):
        """
        Show image with title using OpenCV and Matplotlib.
        
        Args:
            img (numpy.ndarray): Image to show.
            title (str): title of the image.
            color_map (str): Color map to use (default is 'gray').
        """
        # Convert image from BGR (OpenCV default) to RGB if it's not grayscale
        if len(img.shape) == 3:  # Check if image is colored (3 channels)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img, cmap=color_map)
        plt.title(title)
        plt.axis('off')  # Remove axes for cleaner presentation
        plt.show()

    def show_histogram(self, hist, x_label=None, y_label=None, title=None, xlim=None, ylim=None):
        """
        Show histogram using Matplotlib.
        
        Args:
            hist (numpy.ndarray): Histogram data to show.
            x_label (str, optional): Label of the x-axis.
            y_label (str, optional): Label of the y-axis.
            title (str, optional): Title of the histogram.
            xlim (tuple, optional): Limit of the x-axis (min, max).
            ylim (tuple, optional): Limit of the y-axis (min, max).
        """
        plt.plot(hist)
        if title:
            plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.show()

    def show_bar_chart(self, categories, values, title=None, x_label=None, y_label=None):
        """
        Show a bar chart using Matplotlib.
        
        Args:
            categories (list): Categories for the x-axis.
            values (list): Values corresponding to each category.
            title (str, optional): Title of the bar chart.
            x_label (str, optional): Label of the x-axis.
            y_label (str, optional): Label of the y-axis.
        """
        plt.bar(categories, values)
        if title:
            plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        plt.show()

    def show_multiple_images(self, images, labels, cols=2, color_map='gray'):
        """
        Show multiple images in a grid layout using Matplotlib.
        
        Args:
            images (list of numpy.ndarray): List of images to display.
            labels (list of str): List of labels corresponding to each image.
            cols (int): Number of columns in the grid layout.
            color_map (str): Color map to use for images (default is 'gray').
        """
        rows = (len(images) + cols - 1) // cols  # Calculate the required number of rows
        plt.figure(figsize=(12, rows * 4))
        for i, (img, label) in enumerate(zip(images, labels)):
            plt.subplot(rows, cols, i + 1)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img, cmap=color_map)
            plt.title(label)
            plt.axis('off')  # Remove axis for cleaner presentation
        plt.show()
        
        
    def show_image_and_histogram(self, img, hist, image_title='Image', hist_title='Histogram', 
                                x_label='Bins', y_label='Number of Pixels', color_map='gray'):
        """
        Show an image and its corresponding histogram in one line.

        Args:
            img (numpy.ndarray): Image to show.
            hist (numpy.ndarray): Histogram data to show.
            image_title (str): Title for the image.
            hist_title (str): Title for the histogram.
            x_label (str): Label for the x-axis of the histogram.
            y_label (str): Label for the y-axis of the histogram.
            color_map (str): Color map for the image (default is 'gray').
        """
        # Create subplots
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Display the image
        if len(img.shape) == 3:  # Convert BGR to RGB if the image has 3 channels
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[0].imshow(img, cmap=color_map)
        ax[0].set_title(image_title)
        ax[0].axis('off')  # Hide the axis

        # Plot the histogram
        ax[1].plot(hist)
        ax[1].set_title(hist_title)
        ax[1].set_xlabel(x_label)
        ax[1].set_ylabel(y_label)

        plt.tight_layout()
        plt.show()
