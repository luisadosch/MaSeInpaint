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
    Generator to read images.
    """
    for fname in image_files:
        image_path = os.path.join(images_dir, fname)
        try:
            image = Image.open(image_path).convert('RGB')
            yield image
        except Exception as e:
            print(f"Error loading {fname}: {e}")



def preprocess_images(images, target_size=(256, 256)):
    """
    Preprocess images by resizing them and converting to tensors.
    """
    transform = Compose([
        Resize(target_size),
        ToTensor(),
    ])

    image_tensors = [transform(image) for image in images]
    stacked_tensor = torch.stack(image_tensors)
    
    return stacked_tensor



def calculate_fid(real_images_tensor, infilled_images_tensor):
    """
    Calculate the FID score between two sets of images.
    """
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images_tensor, real=True)
    fid.update(infilled_images_tensor, real=False)
    return float(fid.compute())


def bootstrap_fid(real_images_tensor, infilled_images_tensor, n_bootstraps=5):
    """
    Compute bootstrapped FID scores.
    """
    fid_scores = []
    n_images = real_images_tensor.shape[0]
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_images, n_images, replace=True)
        real_sample = real_images_tensor[indices]
        infilled_sample = infilled_images_tensor[indices] 
        fid_value = calculate_fid(real_sample, infilled_sample)
        fid_scores.append(fid_value)
    
    return np.array(fid_scores)


def check_normality(data, alpha=0.05):
    """
    Check the normality of a dataset using the Shapiro–Wilk test.
    """
    data = np.array(data)
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
    """
    data1 = np.array(data1)
    data2 = np.array(data2)
    observed_diff = np.median(data1 - data2)
    stat, p_value = wilcoxon(data1, data2)
    return p_value, observed_diff


def two_sided_t_test(data1, data2, equal_var=False):
    """
    Perform a two-sided t-test comparing the means of two data sets.
    """
    data1 = np.array(data1)
    data2 = np.array(data2)
    observed_diff = np.mean(data1) - np.mean(data2)
    t_stat, p_value = ttest_ind(data1, data2, equal_var=equal_var)
    return p_value, observed_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical tests on the inpainting results.")
    parser.add_argument("-t", "--test", type=str, default="False",
                        help="Set to 'True' to run the tests on a subset of images.")
    parser.add_argument("-l", "--load_data", type=str, default="True",
                        help="Set to 'True' to load bootstrapped numpy files from file server if they exist, else calculate them.")
    args = parser.parse_args()

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler("statistical_tests.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    TEST = args.test.lower() == "true"
    LOAD_DATA = args.load_data.lower() == "true"
    logger.info(f"Running statistical tests (TEST={TEST}, LOAD_DATA={LOAD_DATA})")

    # Directories
    images_dir = './Dataset_new/images'
    masks_dir = './Dataset_new/masks'
    results_dir_lama = 'results/lama'
    results_dir_opencv = 'results/opencv'
    results_dir_sd = 'results/sd'

    real_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') and f in os.listdir(masks_dir)]
    lama_files = [f for f in os.listdir(results_dir_lama) if f.endswith('.jpg')]
    opencv_files = [f for f in os.listdir(results_dir_opencv) if f.endswith('.jpg')]
    sd_files = [f for f in os.listdir(results_dir_sd) if f.endswith('.jpg')]

    image_files = list(set(real_files) & set(lama_files) & set(opencv_files) & set(sd_files))
    logger.info(f"Found {len(image_files)} images in all directories")

    target_size = (256, 256)

    logger.info("Loading the images")
    real_images = list(image_generator(images_dir, image_files))
    lama_images = list(image_generator(results_dir_lama, image_files))
    opencv_images = list(image_generator(results_dir_opencv, image_files))
    sd_images = list(image_generator(results_dir_sd, image_files))

    logger.info(f"Loaded {len(real_images)} Real images")
    logger.info(f"Loaded {len(lama_images)} Lama images")
    logger.info(f"Loaded {len(opencv_images)} OpenCV images")
    logger.info(f"Loaded {len(sd_images)} SD images")

    if TEST:
        real_images = real_images[:100]
        lama_images = lama_images[:100]
        opencv_images = opencv_images[:100]
        sd_images = sd_images[:100]
        N_BOOTSTRAPS = 5
    else:
        N_BOOTSTRAPS = 500

    real_images_tensor = preprocess_images(real_images, target_size)
    lama_images_tensor = preprocess_images(lama_images, target_size)
    opencv_images_tensor = preprocess_images(opencv_images, target_size)
    sd_images_tensor = preprocess_images(sd_images, target_size)

    fid_score_lama = calculate_fid(real_images_tensor, lama_images_tensor)
    fid_score_opencv = calculate_fid(real_images_tensor, opencv_images_tensor)
    fid_score_sd = calculate_fid(real_images_tensor, sd_images_tensor)

    logger.info(f"FID (Lama): {fid_score_lama}")
    logger.info(f"FID (OpenCV): {fid_score_opencv}")
    logger.info(f"FID (SD): {fid_score_sd}")

    fid_scores = {
        "lama": fid_score_lama,
        "opencv": fid_score_opencv,
        "sd": fid_score_sd
    }
    with open("fid_scores.json", "w") as f:
        json.dump(fid_scores, f)
    logger.info("FID scores saved to fid_scores.json")

    logger.info("Calculating bootstrapped FID scores and performing statistical tests")

    # Bootstrapped FID for Lama
    lama_bootstrap_file = "bootstraped_fid_lama.npy"
    if LOAD_DATA and os.path.exists(lama_bootstrap_file):
        bootstraped_fid_lama = np.load(lama_bootstrap_file)
        logger.info(f"Loaded bootstrapped FID Lama from {lama_bootstrap_file}")
    else:
        bootstraped_fid_lama = bootstrap_fid(real_images_tensor, lama_images_tensor, n_bootstraps=N_BOOTSTRAPS)
        np.save(lama_bootstrap_file, bootstraped_fid_lama)
        logger.info(f"Bootstrapped FID Lama computed and saved to {lama_bootstrap_file}")
    logger.info(f"Bootstrapped mean FID (Lama): {np.mean(bootstraped_fid_lama)}, std: {np.std(bootstraped_fid_lama)}")

    # Bootstrapped FID for OpenCV
    opencv_bootstrap_file = "bootstraped_fid_opencv.npy"
    if LOAD_DATA and os.path.exists(opencv_bootstrap_file):
        bootstraped_fid_opencv = np.load(opencv_bootstrap_file)
        logger.info(f"Loaded bootstrapped FID OpenCV from {opencv_bootstrap_file}")
    else:
        bootstraped_fid_opencv = bootstrap_fid(real_images_tensor, opencv_images_tensor, n_bootstraps=N_BOOTSTRAPS)
        np.save(opencv_bootstrap_file, bootstraped_fid_opencv)
        logger.info(f"Bootstrapped FID OpenCV computed and saved to {opencv_bootstrap_file}")
    logger.info(f"Bootstrapped mean FID (OpenCV): {np.mean(bootstraped_fid_opencv)}, std: {np.std(bootstraped_fid_opencv)}")

    # Bootstrapped FID for SD
    sd_bootstrap_file = "bootstraped_fid_sd.npy"
    if LOAD_DATA and os.path.exists(sd_bootstrap_file):
        bootstraped_fid_sd = np.load(sd_bootstrap_file)
        logger.info(f"Loaded bootstrapped FID SD from {sd_bootstrap_file}")
    else:
        bootstraped_fid_sd = bootstrap_fid(real_images_tensor, sd_images_tensor, n_bootstraps=N_BOOTSTRAPS)
        np.save(sd_bootstrap_file, bootstraped_fid_sd)
        logger.info(f"Bootstrapped FID SD computed and saved to {sd_bootstrap_file}")
    logger.info(f"Bootstrapped mean FID (SD): {np.mean(bootstraped_fid_sd)}, std: {np.std(bootstraped_fid_sd)}")

    logger.info("Normality test (Lama)")
    normal_lama, details_lama = check_normality(bootstraped_fid_lama)
    logger.info(f"Normality test (Lama): {normal_lama}, details: {details_lama}")

    logger.info("Normality test (OpenCV)")
    normal_opencv, details_opencv = check_normality(bootstraped_fid_opencv)
    logger.info(f"Normality test (OpenCV): {normal_opencv}, details: {details_opencv}")

    logger.info("Normality test (SD)")
    normal_sd, details_sd = check_normality(bootstraped_fid_sd)
    logger.info(f"Normality test (SD): {normal_sd}, details: {details_sd}")

    logger.info("Performing statistical tests")

    logger.info("Wilcoxon signed-rank test (OpenCV vs Lama)")
    p_value_opencv_lama, observed_diff_opencv_lama = wilcoxon_signed_rank_test(bootstraped_fid_opencv, bootstraped_fid_lama)
    logger.info(f"p-value (OpenCV vs Lama): {p_value_opencv_lama}, observed difference: {observed_diff_opencv_lama}")

    logger.info("Wilcoxon signed-rank test (SD vs Lama)")
    p_value_sd_lama, observed_diff_sd_lama = wilcoxon_signed_rank_test(bootstraped_fid_sd, bootstraped_fid_lama)
    logger.info(f"p-value (SD vs Lama): {p_value_sd_lama}, observed difference: {observed_diff_sd_lama}")

    logger.info("Wilcoxon signed-rank test (Lama vs SD)")
    p_value_lama_sd, observed_diff_lama_sd = wilcoxon_signed_rank_test(bootstraped_fid_lama, bootstraped_fid_sd)
    logger.info(f"p-value (Lama vs SD): {p_value_lama_sd}, observed difference: {observed_diff_lama_sd}")

    logger.info("Wilcoxon signed-rank test (OpenCV vs SD)")
    p_value_opencv_sd, observed_diff_opencv_sd = wilcoxon_signed_rank_test(bootstraped_fid_opencv, bootstraped_fid_sd)
    logger.info(f"p-value (OpenCV vs SD): {p_value_opencv_sd}, observed difference: {observed_diff_opencv_sd}")

    logger.info("Two-sided t-test (OpenCV vs Lama)")
    p_value_opencv_lama_t, observed_diff_opencv_lama_t = two_sided_t_test(bootstraped_fid_opencv, bootstraped_fid_lama)
    logger.info(f"p-value (OpenCV vs Lama): {p_value_opencv_lama_t}, observed difference: {observed_diff_opencv_lama_t}")

    logger.info("Two-sided t-test (SD vs Lama)")
    p_value_sd_lama_t, observed_diff_sd_lama_t = two_sided_t_test(bootstraped_fid_sd, bootstraped_fid_lama)
    logger.info(f"p-value (SD vs Lama): {p_value_sd_lama_t}, observed difference: {observed_diff_sd_lala_t if 'observed_diff_sd_lala_t' in locals() else observed_diff_sd_lama_t}")

    logger.info("Two-sided t-test (Lama vs SD)")
    p_value_lama_sd_t, observed_diff_lama_sd_t = two_sided_t_test(bootstraped_fid_lama, bootstraped_fid_sd)
    logger.info(f"p-value (Lama vs SD): {p_value_lama_sd_t}, observed difference: {observed_diff_lama_sd_t}")

    logger.info("Two-sided t-test (OpenCV vs SD)")
    p_value_opencv_sd_t, observed_diff_opencv_sd_t = two_sided_t_test(bootstraped_fid_opencv, bootstraped_fid_sd)
    logger.info(f"p-value (OpenCV vs SD): {p_value_opencv_sd_t}, observed difference: {observed_diff_opencv_sd_t}")

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
