#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import numpy as np
import os
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
from PIL import Image
import os
import torch
from torchvision.transforms import ToTensor, Resize, Compose
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
from PIL import Image
import numpy as np

import logging
import json


def image_generator(images_dir, image_files):
    """
    Generator to read images and their corresponding masks.

    Args:
    - images_dir (str): Directory containing the images.
    - masks_dir (str): Directory containing the masks.
    - image_files (list): List of filenames (same for both images and masks).

    Yields:
    - tuple: A tuple containing a PIL image and a corresponding mask (PIL image).
    """
    for fname in image_files:
        image_path = os.path.join(images_dir, fname)  # Full path to the image
        
        # Load image and mask
        try:
            image = Image.open(image_path).convert('RGB')  # Convert image to RGB
            
            # Yield image and mask
            yield image
        except Exception as e:
            print(f"Error loading {fname}: {e}")



# Preprocessing function to resize and convert images to tensors
def preprocess_images(images, target_size=(256, 256)):
    """
    Preprocess images by resizing them to the target size and converting to tensors.

    Args:
    - images (list of PIL Images): The list of images to preprocess.
    - target_size (tuple): The target size to resize images to (default is 256x256).

    Returns:
    - torch.Tensor: A tensor containing all preprocessed images, stacked in a batch.
    """
    transform = Compose([
        Resize(target_size),  # Resize the images to the target size
        ToTensor(),  # Convert the images to tensor
    ])

    # Apply the transformation to each image in the list
    image_tensors = [transform(image) for image in images]

    # Stack the tensors into a single batch
    stacked_tensor = torch.stack(image_tensors)
    
    return stacked_tensor



# Error calculation function to compute the FID score
def calculate_fid(real_images_tensor, infilled_images_tensor):
    """
    Calculate the Fréchet Inception Distance (FID) between real and Lama generated images.

    Args:
    - real_images_tensor (tensor): The tensor of real images.
    - infilled_images_tensor (list of PIL Images): The tensor of infilled images.

    Returns:
    - float: The FID score.
    """
    # Initialize the FID metric
    fid = FrechetInceptionDistance(normalize=True)

    # Update the FID with the preprocessed images
    fid.update(real_images_tensor, real=True)
    fid.update(infilled_images_tensor, real=False)

    # Calculate and return the FID score
    return float(fid.compute())


def bootstrap_fid(real_images_tensor, infilled_images_tensor, n_bootstraps=5):
    """
    Compute bootstrapped FID scores and return the mean FID,
    95% confidence interval, and the full distribution of scores.

    Args:
    - real_images_tensor (tensor): Tensor of real images (shape: [N, C, H, W]).
    - infilled_images_tensor (list of PIL Images): List of generated images.
    - n_bootstraps (int): Number of bootstrap iterations.

    Returns:
    - mean_fid (float): Mean FID score across bootstrap samples.
    - ci (tuple): 95% confidence interval (lower, upper) of the FID.
    - fid_scores (np.array): Array of all bootstrap FID scores.
    """
    fid_scores = []
    n_real = real_images_tensor.shape[0]       # Assuming batch dimension is the first
    n_infilled = len(infilled_images_tensor)     # Number of generated images
    
    for _ in range(n_bootstraps):
        # Sample indices with replacement for real images
        real_indices = np.random.choice(n_real, n_real, replace=True)
        # Sample indices with replacement for generated images
        infilled_indices = np.random.choice(n_infilled, n_infilled, replace=True)
        
        # Create bootstrap samples
        real_sample = real_images_tensor[real_indices]
        infilled_sample = infilled_images_tensor[infilled_indices] 
        
        # Compute the FID score for these samples
        fid_value = calculate_fid(real_sample, infilled_sample)
        fid_scores.append(fid_value)
    
    fid_scores = np.array(fid_scores)
    
    return fid_scores


def permutation_test(data1, data2, n_permutations=10000):
    """
    Perform a permutation test comparing the means of two data sets.
    
    Args:
      data1 (array-like): Bootstrapped FID scores for method 1.
      data2 (array-like): Bootstrapped FID scores for method 2.
      n_permutations (int): Number of permutations to perform.
      
    Returns:
      p_value (float): p-value from the permutation test.
      observed_diff (float): The observed difference in means.
    """
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    observed_diff = np.mean(data1) - np.mean(data2)
    
    # Combine the data from both groups
    combined = np.concatenate([data1, data2])
    
    count = 0
    for _ in range(n_permutations):
        # Shuffle the combined data
        np.random.shuffle(combined)
        # Split the data into two groups with the same sizes as the original groups
        perm_data1 = combined[:len(data1)]
        perm_data2 = combined[len(data1):]
        # Compute the difference in means for this permutation
        perm_diff = np.mean(perm_data1) - np.mean(perm_data2)
        if np.abs(perm_diff) >= np.abs(observed_diff):
            count += 1

    p_value = count / n_permutations
    return p_value, observed_diff



import logging

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("statistical_tests.log")
file_handler.setLevel(logging.INFO)

# Create formatters and add them to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)



# Directories
images_dir = './Dataset_new/images'
results_dir_lama = 'results/lama'
results_dir_opencv = 'results/opencv'
results_dir_sd = 'results/sd'

# List all files in images directory    
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
lama_files = [f for f in os.listdir(results_dir_lama) if f.endswith('.jpg')]
opencv_files = [f for f in os.listdir(results_dir_opencv) if f.endswith('.jpg')]
sd_files = [f for f in os.listdir(results_dir_sd) if f.endswith('.jpg')]

# Define a fixed image size
target_size = (256, 256)  # Resize all images to this size


# Define

# Load the images
logger.info("Loading the images")

real_images = list(image_generator(images_dir, image_files))
lama_images = list(image_generator(results_dir_lama, lama_files))
opencv_images = list(image_generator(results_dir_opencv, opencv_files))
sd_images = list(image_generator(results_dir_sd, sd_files))

logger.info(f"Loaded {len(real_images)} real images")

TEST = False

# If TEST is True, only use the first 100 images for testing
if TEST:
    real_images = real_images[:100]
    lama_images = lama_images[:100]
    opencv_images = opencv_images[:100]
    sd_images = sd_images[:100]

    N_BOOTSTRAPS = 5
else:
    N_BOOTSTRAPS = 50

# Preprocess the images (resize and convert to tensors)
real_images_tensor = preprocess_images(real_images, target_size)
lama_images_tensor = preprocess_images(lama_images, target_size)
opencv_images_tensor = preprocess_images(opencv_images, target_size)
sd_images_tensor = preprocess_images(sd_images, target_size)


# Calculate the FID score between the real and generated images

fid_score_lama = calculate_fid(real_images_tensor, lama_images_tensor)
fid_score_opencv = calculate_fid(real_images_tensor, opencv_images_tensor)
fid_score_sd = calculate_fid(real_images_tensor, sd_images_tensor)

logger.info(f"FID (Lama): {fid_score_lama}")
logger.info(f"FID (OpenCV): {fid_score_opencv}")
logger.info(f"FID (SD): {fid_score_sd}")

#Store the FID scores in a dictionar and save the dictionary to a file
fid_scores = {
    "lama": fid_score_lama,
    "opencv": fid_score_opencv,
    "sd": fid_score_sd
}

with open("fid_scores.json", "w") as f:
    json.dump(fid_scores, f)

logger.info("FID scores saved to fid_scores.json")

logger.info("Calculating bootstrapped FID scores and performing permutation tests")

# Calculate bootstrapped FID scores for Lama, OpenCV, and SD

logger.info("Calculating bootstrapped FID Lama")
bootstraped_fid_lama = bootstrap_fid(real_images_tensor, lama_images_tensor, n_bootstraps=N_BOOTSTRAPS)
# Save the bootstrapped FID scores to a file
np.save("bootstraped_fid_lama.npy", bootstraped_fid_lama)
logger.info("Bootstrapped FID scores saved to bootstraped_fid_lama.npy")



logger.info("Calculating bootstrapped OpenCV")
bootstraped_fid_opencv = bootstrap_fid(real_images_tensor, opencv_images_tensor, n_bootstraps=N_BOOTSTRAPS)
np.save("bootstraped_fid_opencv.npy", bootstraped_fid_opencv)
logger.info("Bootstrapped FID scores saved to bootstraped_fid_opencv.npy")

logger.info("Permutation test (Lama vs. OpenCV)")
p_value_lama_opencv, observed_diff_lama_opencv = permutation_test(bootstraped_fid_lama, bootstraped_fid_opencv)
logger.info(f"Permutation test (Lama vs. OpenCV): p-value = {p_value_lama_opencv}, observed difference = {observed_diff_lama_opencv}")

logger.info("Calculating bootstrapped SD Inpaint")
bootstraped_fid_sd = bootstrap_fid(real_images_tensor, sd_images_tensor, n_bootstraps=N_BOOTSTRAPS)
np.save("bootstraped_fid_sd.npy", bootstraped_fid_sd)
logger.info("Bootstrapped FID scores saved to bootstraped_fid_sd.npy")

logger.info("Permutation test (Lama vs. SD)")
p_value_lama_sd, observed_diff_lama_sd = permutation_test(bootstraped_fid_lama, bootstraped_fid_sd)
logger.info(f"Permutation test (Lama vs. SD): p-value = {p_value_lama_sd}, observed difference = {observed_diff_lama_sd}")

logger.info("Permutation test (OpenCV vs. SD)")
p_value_opencv_sd, observed_diff_opencv_sd = permutation_test(bootstraped_fid_opencv, bootstraped_fid_sd)
logger.info(f"Permutation test (OpenCV vs. SD): p-value = {p_value_opencv_sd}, observed difference = {observed_diff_opencv_sd}")


# Store the permutation test results in a dictionary and save to a file
permutation_test_results = {
    "lama_vs_opencv": {
        "p_value": p_value_lama_opencv,
        "observed_diff": observed_diff_lama_opencv
    },
    "lama_vs_sd": {
        "p_value": p_value_lama_sd,
        "observed_diff": observed_diff_lama_sd
    },
    "opencv_vs_sd": {
        "p_value": p_value_opencv_sd,
        "observed_diff": observed_diff_opencv_sd
    }
}

with open("permutation_test_results.json", "w") as f:
    json.dump(permutation_test_results, f)


