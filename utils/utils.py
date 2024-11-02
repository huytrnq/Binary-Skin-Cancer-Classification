import json
import os
import cv2
import shutil
import pickle
import numpy as np
from datetime import datetime
from IPython.display import display, Javascript

def export_experiment(name, params, feature_dict, model, notebook_name, output_folder="experiments"):
    """
    Export experiment data to a specified folder with parameters and features.
    
    Args:
        name (str): Name of the experiment.
        params (dict): Nested dictionary of experiment parameters for each descriptor.
        feature_dict (dict): Dictionary of training and testing features.
        model (object): Trained model object with full pipeline.
        notebook_name (str): Name of the notebook that ran the experiment.
        output_folder (str): Directory to store experiment files.
    """
    # Create a timestamped folder within the output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = os.path.join(output_folder, f"{name}_{timestamp}")
    os.makedirs(experiment_folder, exist_ok=True)

    # Save parameters to a JSON file with nested structure
    params_file = os.path.join(experiment_folder, "params.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=4)

    # Save features and labels as a .npy file
    for key, value in feature_dict.items():
        np.save(os.path.join(experiment_folder, f"{key}.npy"), value)
        
    # Save the model object as a .pkl file
    model_file = os.path.join(experiment_folder, "model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    
    # Save the notebook name that ran the experiment
    display(Javascript(f"IPython.notebook.save_notebook()"))
    notebook_src = os.path.join(os.getcwd(), notebook_name)
    notebook_dest = os.path.join(experiment_folder, notebook_name)
    shutil.copy(notebook_src, notebook_dest)
    
    print(f"Experiment '{name}' saved at {experiment_folder}")
    

def load_experiment(experiment_folder):
    """
    Load experiment data from a specified folder.
    
    Args:
        experiment_folder (str): Directory containing the experiment files.
        
    Returns:
        params (dict): Nested dictionary of experiment parameters for each descriptor.
        feature_dict (dict): Dictionary of training and testing features.
        model (object): Trained model object with full pipeline.
    """
    # Load parameters from a JSON file
    params_file = os.path.join(experiment_folder, "params.json")
    with open(params_file, "r") as f:
        params = json.load(f)
        
    # Load features and labels from .npy files
    feature_dict = {}
    for file in os.listdir(experiment_folder):
        if file.endswith(".npy"):
            key = file.split(".")[0]
            feature_dict[key] = np.load(os.path.join(experiment_folder, file))
    
    # Load the model object from a .pkl file
    model_file = os.path.join(experiment_folder, "model.pkl")
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    
    return params, feature_dict, model
    

def sliding_window(image, window_size, step_size):
    """
    A generator that yields the coordinates and the window of the image.
    
    Args:
        image (numpy array): The input image (grayscale or color).
        window_size (tuple): The size of the window (height, width).
        step_size (int): The step size for sliding the window.
        
    Yields:
        (x, y, window): The top-left corner (x, y) and the window (region) of the image.
    """
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
            
            


def distance_transform_weighting(image, mask, max_distance=50, weight=0.3):
    """
    Applies a distance-based weighting on the mask to smoothly transition from the ROI to the background.

    Args:
        image (numpy array): Input image.
        mask (numpy array): Binary mask (1 for ROI, 0 for background).
        max_distance (int): Maximum distance for weighting.

    Returns:
        weighted_image (numpy array): Image with distance-weighted mask.
    """
    # Calculate distance transform of the mask
    dist_transform = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)

    # Normalize distance to range [0, 1] and apply max distance
    normalized_dist = np.clip(dist_transform / max_distance, 0, 1)
    weight_mask = 1 - normalized_dist  # Higher weight near the ROI

    # Repeat the weight mask across color channels
    weight_mask = np.repeat(weight_mask[:, :, np.newaxis], 3, axis=2)
    
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Blend the image with the weighted mask
    weighted_image = image * weight_mask + image * (1 - weight_mask) * weight
    
    # Ensure mask is single-channel and 8-bit unsigned
    weighted_image = weighted_image.astype(np.uint8)
    weighted_image = cv2.cvtColor(weighted_image, cv2.COLOR_BGR2GRAY) if weighted_image.ndim == 3 else weighted_image

    return weighted_image




def weighted_mask(image, mask, blur_size=51):
    """
    Applies a weighted mask to retain some background context by softening the mask edges.

    Args:
        image (numpy array): Input image.
        mask (numpy array): Binary mask (1 for ROI, 0 for background).
        blur_size (int): Kernel size for Gaussian blur to soften mask edges (must be positive and odd).

    Returns:
        weighted_image (numpy array): Image with weighted mask applied.
    """
    # Ensure blur_size is positive and odd
    if blur_size <= 0:
        blur_size = 1
    elif blur_size % 2 == 0:
        blur_size += 1
        
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert the mask to a float (0 to 1 range)
    soft_mask = mask.astype(float)
    
    # Apply Gaussian blur to soften the edges of the mask
    soft_mask = cv2.GaussianBlur(soft_mask, (blur_size, blur_size), 0)

    # Expand dimensions to match the image channels (3 for color image)
    soft_mask = np.repeat(soft_mask[:, :, np.newaxis], 3, axis=2)

    # Apply the weighted mask to the image
    weighted_image = image * soft_mask + image * (1 - soft_mask) * 0.5  # retain some context from background
    
    # Ensure mask is single-channel and 8-bit unsigned
    weighted_image = weighted_image.astype(np.uint8)
    weighted_image = cv2.cvtColor(weighted_image, cv2.COLOR_BGR2GRAY) if weighted_image.ndim == 3 else weighted_image

    return weighted_image
        