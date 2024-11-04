import json
import os
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
            
        