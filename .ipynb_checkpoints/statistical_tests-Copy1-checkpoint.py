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
from scipy.stats import wilcoxon, ttest_ind, shapiro

import argparse



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
    - infilled_images_tensor (tensor): Tensor of infilled images (shape: [N, C, H, W]).
    - n_bootstraps (int): Number of bootstrap iterations.

    Returns:
    - np.array: Array of bootstrapped FID scores.
    """
    fid_scores = []
    n_images = real_images_tensor.shape[0]       # Assuming batch dimension is the first
    
    for _ in range(n_bootstraps):
        # Sample indices with replacement for real images
        indices = np.random.choice(n_images, n_images, replace=True)
        
        # Create bootstrap samples
        real_sample = real_images_tensor[indices]
        infilled_sample = infilled_images_tensor[indices] 
        
        # Compute the FID score for these samples
        fid_value = calculate_fid(real_sample, infilled_sample)
        fid_scores.append(fid_value)
    
    fid_scores = np.array(fid_scores)
    
    return fid_scores


def check_normality(data, alpha=0.05):
    """
    Check the normality assumption for a data set using the Shapiro–Wilk test.
    
    Args:
      data (array-like): Data for the test.
      alpha (float): Significance level for the test (default is 0.05).
      
    Returns:
      normal (bool): True if the data is normally distributed.
      details (dict): Detailed test statistics and p-value.
    """
    data = np.array(data)
    
    # Perform the Shapiro-Wilk test
    stat, p_value = shapiro(data)
    normal = p_value > alpha
    
    details = {
        'statistic': stat,
        'p_value': p_value,
        'normal': str(normal)
    }
    
    return normal, details

def wilcoxon_signed_rank_test(data1, data2):
    """
    Perform a Wilcoxon signed-rank test comparing paired data sets.
    
    Args:
      data1 (array-like): Paired bootstrapped FID scores for method 1.
      data2 (array-like): Paired bootstrapped FID scores for method 2.
      
    Returns:
      p_value (float): p-value from the Wilcoxon signed-rank test (two-sided).
      observed_diff (float): The median difference in scores (data1 - data2).
    """
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    # Compute the observed median difference
    observed_diff = np.median(data1 - data2)
    
    # Perform the Wilcoxon signed-rank test
    # Note: By default, scipy.stats.wilcoxon performs a two-sided test.
    stat, p_value = wilcoxon(data1, data2)
    
    return p_value, observed_diff


def two_sided_t_test(data1, data2, equal_var=False):
    """
    Perform a two-sided t-test comparing the means of two data sets.
    
    Args:
      data1 (array-like): Bootstrapped FID scores for method 1.
      data2 (array-like): Bootstrapped FID scores for method 2.
      equal_var (bool): Whether to assume equal population variances. Default is True.
      
    Returns:
      p_value (float): p-value from the two-sided t-test.
      observed_diff (float): The observed difference in means (data1 - data2).
    """
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    observed_diff = np.mean(data1) - np.mean(data2)
    t_stat, p_value = ttest_ind(data1, data2, equal_var=equal_var)
    
    return p_value, observed_diff


import logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical tests on the inpainting results.")
    parser.add_argument("-t", "--test", type=str, default="False", help="Set to 'True' to run the tests on a subset of images.")
    args = parser.parse_args()

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

    # Convert string input to boolean
    TEST = args.test.lower() == "true"

    logger.info(f"Running statistical tests (TEST={TEST})")

    # Directories
    images_dir = './Dataset_new/images'
    masks_dir = './Dataset_new/masks'
    results_dir_lama = 'results/lama'
    results_dir_opencv = 'results/opencv'
    results_dir_sd = 'results/sd'

    # List all files in images directory    
    real_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') and f in os.listdir(masks_dir)]
    lama_files = [f for f in os.listdir(results_dir_lama) if f.endswith('.jpg')]
    opencv_files = [f for f in os.listdir(results_dir_opencv) if f.endswith('.jpg')]
    sd_files = [f for f in os.listdir(results_dir_sd) if f.endswith('.jpg')]

    # Get all file names that are in all three directories
    image_files = list(set(real_files) & set(lama_files) & set(opencv_files) & set(sd_files))

    logger.info(f"Found {len(image_files)} images in all directories")

    # Define a fixed image size
    target_size = (256, 256)  # Resize all images to this size


    # Define

    # Load the images
    logger.info("Loading the images")

    real_images = list(image_generator(images_dir, image_files))
    lama_images = list(image_generator(results_dir_lama, image_files))
    opencv_images = list(image_generator(results_dir_opencv, image_files))
    sd_images = list(image_generator(results_dir_sd, image_files))

    logger.info(f"Loaded {len(real_images)} Real images")
    logger.info(f"Loaded {len(lama_images)} Lama images")
    logger.info(f"Loaded {len(opencv_images)} OpenCV images")
    logger.info(f"Loaded {len(sd_images)} SD images")



    # If TEST is True, only use the first 100 images for testing
    if TEST:
        real_images = real_images[:100]
        lama_images = lama_images[:100]
        opencv_images = opencv_images[:100]
        sd_images = sd_images[:100]

        N_BOOTSTRAPS = 5
    else:
        N_BOOTSTRAPS = 500

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

    logger.info("Calculating bootstrapped FID scores and performing statistical tests")

    # Calculate bootstrapped FID scores for Lama, OpenCV, and SD

    logger.info("Calculating bootstrapped FID Lama")
    bootstraped_fid_lama = bootstrap_fid(real_images_tensor, lama_images_tensor, n_bootstraps=N_BOOTSTRAPS)
    # Save the bootstrapped FID scores to a file
    np.save("bootstraped_fid_lama.npy", bootstraped_fid_lama)
    logger.info("Bootstrapped FID scores saved to bootstraped_fid_lama.npy")
    logger.info(f"Bootstrapped mean FID (Lama): {np.mean(bootstraped_fid_lama)}, std: {np.std(bootstraped_fid_lama)}")

    # Check if the bootstrapped FID scores are normally distributed
    logger.info("Normality test (Lama)")
    normal_lama, details_lama = check_normality(bootstraped_fid_lama)
    logger.info(f"Normality test (Lama): {normal_lama}, details: {details_lama}")

    logger.info("Calculating bootstrapped OpenCV")
    bootstraped_fid_opencv = bootstrap_fid(real_images_tensor, opencv_images_tensor, n_bootstraps=N_BOOTSTRAPS)
    np.save("bootstraped_fid_opencv.npy", bootstraped_fid_opencv)
    logger.info("Bootstrapped FID scores saved to bootstraped_fid_opencv.npy")
    logger.info(f"Bootstrapped mean FID (OpenCV): {np.mean(bootstraped_fid_opencv)}, std: {np.std(bootstraped_fid_opencv)}")

    # Check if the bootstrapped FID scores are normally distributed
    logger.info("Normality test (OpenCV)")
    normal_opencv, details_opencv = check_normality(bootstraped_fid_opencv)
    logger.info(f"Normality test (OpenCV): {normal_opencv}, details: {details_opencv}")


    logger.info("Calculating bootstrapped SD Inpaint")
    bootstraped_fid_sd = bootstrap_fid(real_images_tensor, sd_images_tensor, n_bootstraps=N_BOOTSTRAPS)
    np.save("bootstraped_fid_sd.npy", bootstraped_fid_sd)
    logger.info("Bootstrapped FID scores saved to bootstraped_fid_sd.npy")
    logger.info(f"Bootstrapped mean FID (SD): {np.mean(bootstraped_fid_sd)}, std: {np.std(bootstraped_fid_sd)}")


    # Check if the bootstrapped FID scores are normally distributed
    logger.info("Normality test (SD)")
    normal_sd, details_sd = check_normality(bootstraped_fid_sd)
    logger.info(f"Normality test (SD): {normal_sd}, details: {details_sd}")

    # Perform statistical tests
    logger.info("Performing statistical tests")

    # Perform Wilcoxon signed-rank test between Lama and OpenCV
    logger.info("Wilcoxon signed-rank test (OpenCV vs Lama)")
    p_value_opencv_lama, observed_diff_opencv_lama = wilcoxon_signed_rank_test(bootstraped_fid_opencv, bootstraped_fid_lama)
    logger.info(f"p-value (OpenCV vs Lama): {p_value_opencv_lama}, observed difference: {observed_diff_opencv_lama}")

    # Perform Wilcoxon signed-rank test between Lama and SD
    logger.info("Wilcoxon signed-rank test (SD vs Lama)")
    p_value_sd_lama, observed_diff_sd_lama = wilcoxon_signed_rank_test(bootstraped_fid_sd, bootstraped_fid_lama)
    logger.info(f"p-value (SD vs Lama): {p_value_sd_lama}, observed difference: {observed_diff_sd_lama}")

    logger.info("Wilcoxon signed-rank test (Lama vs SD)")
    p_value_lama_sd, observed_diff_lama_sd = wilcoxon_signed_rank_test(bootstraped_fid_lama, bootstraped_fid_sd)
    logger.info(f"p-value (Lama v): {p_value_lama_sd}, observed difference: {observed_diff_lama_sd}")

    # Perform Wilcoxon signed-rank test between OpenCV and SD
    logger.info("Wilcoxon signed-rank test (OpenCV vs SD)")
    p_value_opencv_sd, observed_diff_opencv_sd = wilcoxon_signed_rank_test(bootstraped_fid_opencv, bootstraped_fid_sd)
    logger.info(f"p-value (OpenCV vs SD): {p_value_opencv_sd}, observed difference: {observed_diff_opencv_sd}")

    # Perform two-sided t-test between Lama and OpenCV
    logger.info("Two-sided t-test (OpenCV vs Lama)")
    p_value_opencv_lama_t, observed_diff_opencv_lama_t = two_sided_t_test(bootstraped_fid_opencv, bootstraped_fid_lama)
    logger.info(f"p-value (OpenCV vs Lama): {p_value_opencv_lama_t}, observed difference: {observed_diff_opencv_lama_t}")

    # Perform two-sided t-test between Lama and SD
    logger.info("Two-sided t-test (SD vs Lama)")
    p_value_sd_lama_t, observed_diff_sd_lama_t = two_sided_t_test(bootstraped_fid_sd, bootstraped_fid_lama)
    logger.info(f"p-value (SD vs Lama): {p_value_sd_lama_t}, observed difference: {observed_diff_sd_lama_t}")

    logger.info("Two-sided t-test (Lama vs SD)")
    p_value_lama_sd_t, observed_diff_lama_sd_t = two_sided_t_test(bootstraped_fid_lama, bootstraped_fid_sd)
    logger.info(f"p-value (Lama vs SD): {p_value_lama_sd_t}, observed difference: {observed_diff_lama_sd_t}")

    # Perform two-sided t-test between OpenCV and SD
    logger.info("Two-sided t-test (OpenCV vs SD)")
    p_value_opencv_sd_t, observed_diff_opencv_sd_t = two_sided_t_test(bootstraped_fid_opencv, bootstraped_fid_sd)
    logger.info(f"p-value (OpenCV vs SD): {p_value_opencv_sd_t}, observed difference: {observed_diff_opencv_sd_t}")

    # Save the results to a file alltests (including normality tests)
    results = {
        "bootstrapped_fid": {
            "lama": {"mean": np.mean(bootstraped_fid_lama), "std": np.std(bootstraped_fid_lama)},
            "opencv": {"mean": np.mean(bootstraped_fid_opencv), "std": np.std(bootstraped_fid_opencv)},
            "sd": {"mean": np.mean(bootstraped_fid_sd), "std": np.std(bootstraped_fid_sd)}
        },
        "normality": {
            "lama":  details_lama,
            "opencv":  details_opencv,
            "sd": details_sd,
        },
        "wilcoxon": {
            "opencv_lama": {
                "p_value": p_value_opencv_lama,
                "observed_diff": observed_diff_opencv_lama
            },
            "sd_lama": {
                "p_value": p_value_sd_lama,
                "observed_diff": observed_diff_sd_lama
            },
            "lama_sd": {
                "p_value": p_value_lama_sd,
                "observed_diff": observed_diff_lama_sd
            },
            "opencv_sd": {
                "p_value": p_value_opencv_sd,
                "observed_diff": observed_diff_opencv_sd
            }
        },
        "t_test": {
            "opencv_lama": {
                "p_value": p_value_opencv_lama_t,
                "observed_diff": observed_diff_opencv_lama_t
            },
            "sd_lama": {
                "p_value": p_value_sd_lama_t,
                "observed_diff": observed_diff_sd_lama_t
            },
            "lama_sd": {
                "p_value": p_value_lama_sd_t,
                "observed_diff": observed_diff_lama_sd_t
            },
            "opencv_sd": {
                "p_value": p_value_opencv_sd_t,
                "observed_diff": observed_diff_opencv_sd_t
            }
        }
    }

    with open("statistical_tests.json", "w") as f:
        json.dump(results, f)




