import os
import json
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.dataloader import DataLoader
from utils.vis import MatplotlibVisualizer
from utils.transforms import HairRemoval, Composer, CentricCropping
from utils.utils import export_experiment
from descriptors.shape import HOGDescriptor
from utils.segmentation import ThresholdingSegmentation
from descriptors.stats import IntensityStatsGridDescriptor
from descriptors.texture import LBPDescriptor, GLCMDescriptor, GaborFilterDescriptor, ColorMultiScaleLBPDescriptor
from descriptors.color import ColorDescriptor, ColorLayoutDescriptor, ColorCooccurrenceMatrixDescriptor

## Classes
CLASSES = ['nevus', 'others']

## Work folder
work_folder = os.getcwd()
data_folder = os.path.join(work_folder, '..', 'Data/Challenge1')
features_folder = os.path.join(data_folder, 'features')

os.makedirs(features_folder, exist_ok=True)

## Visualizer
matplotlib_visualizer = MatplotlibVisualizer()
exp_name = 'binary_classification'

## Define parameters
params = {
    'color_layout': {
        'grid_x': 1,
        'grid_y': 1,
    },
    'intensity_stats': {
        'grid_x': 1,
        'grid_y': 1,
    },
    'color': {
        'bins': (8, 12, 3),
    },
    'glcm': {
        'distances': [1],
        'angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],
        'levels': 8,
        'grid_x': 1,
        'grid_y': 1,
    },
    'lbp': {
        'radius': 3,
        'n_points': 16,
        'grid_x': 1,
        'grid_y': 1,
    },
    'color_cooccurrence_matrix': {
        'distances': [1],
        'angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],
        'levels': 8,
        'grid_x': 1,
        'grid_y': 1,
    },
    'color_multiscale_lbp': {
        'radius': 3,
        'n_points': 16,
        'scales': [1, 2],
    },
}

modes = ['train', 'val']

## Descriptors
color_layout_descriptor = ColorLayoutDescriptor(**params['color_layout'])
intensity_stats_grid_descriptor = IntensityStatsGridDescriptor(**params['intensity_stats'])
color_descriptor = ColorDescriptor(**params['color'])
glcm_descriptor = GLCMDescriptor(**params['glcm'])
lbp_descriptor = LBPDescriptor(**params['lbp'])
color_cooccurrence_matrix_descriptor = ColorCooccurrenceMatrixDescriptor(**params['color_cooccurrence_matrix'])
color_multiscale_lbp_descriptor = ColorMultiScaleLBPDescriptor(**params['color_multiscale_lbp'])

for mode in modes:

    ## Data loader
    max_samples = 100  # Set a limit for training samples
    balance = False  # Balance dataset if needed
    dataloader = DataLoader(data_folder, mode, 
                            shuffle=False, 
                            ignore_folders=['black_background', '.DS_Store'], 
                            max_samples=max_samples, 
                            balance=balance,
                            transforms=None, 
                            classes=CLASSES, 
                            mask=False)

    # Initialize dictionaries to store features for each type
    features = {
        'color': [],
        'color_layout': [],
        'intensity_stats': [],
        'glcm': [],
        'lbp': [],
        'color_cooccurrence_matrix': [],
        'color_multiscale_lbp': []
    }
    labels = []

    for i, (img, label, mask, path) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Extracting features for {mode}'):
        ## Extract and save features for each descriptor
        features['color'].append(color_descriptor.extract(img))
        features['color_layout'].append(color_layout_descriptor.extract(img))
        features['intensity_stats'].append(intensity_stats_grid_descriptor.extract(img))
        features['glcm'].append(glcm_descriptor.extract(img))
        features['lbp'].append(lbp_descriptor.extract(img))
        features['color_cooccurrence_matrix'].append(color_cooccurrence_matrix_descriptor.extract(img))
        features['color_multiscale_lbp'].append(color_multiscale_lbp_descriptor.extract(img))
        
        labels.append(label)
    
    import pdb; pdb.set_trace()
    ## Save features separately
    os.makedirs(os.path.join(features_folder, mode), exist_ok=True)
    for feature_name, feature_data in features.items():
        np.save(os.path.join(features_folder, mode, feature_name + '.npy'), feature_data)
    
    ## Save labels
    np.save(os.path.join(features_folder, mode, 'labels.npy'), labels)

## Export json file with parameters
with open(os.path.join(features_folder, 'params.json'), 'w') as f:
    json.dump(params, f)
