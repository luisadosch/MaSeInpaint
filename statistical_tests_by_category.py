import pandas as pd
import json
import os
import torch
from statistical_tests import image_generator, preprocess_images, calculate_fid, bootstrap_fid, check_normality, wilcoxon_signed_rank_test, two_sided_t_test
import logging
import numpy as np
import argparse

def get_mask_array(images_tensor, df_mask, category, image_files):
    """
    Filtert die Bilder basierend auf ihrer Maskenkategorie.
    
    Args:
    - images_tensor (torch.Tensor): Tensor der Bilder.
    - df_mask (pd.DataFrame): DataFrame mit den Maskenkategorien.
    - category (str): Die Kategorie ("klein", "mittel", "groß").
    
    Returns:
    - torch.Tensor: Tensor der gefilterten Bilder.
    """
    # Dateinamen der Bilder in der Kategorie
    category_files = df_mask[df_mask["mask_category"] == category]["Bildname"].tolist()
    
    # Indizes der Bilder in der Kategorie
    indices = [i for i, fname in enumerate(image_files) if fname in category_files and i < images_tensor.shape[0]]
    
    # Wähle die entsprechenden Bilder aus
    result = images_tensor[indices]
    
    return result

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run statistical tests on the inpainting results.")
    parser.add_argument("-t", "--test", type=str, default="False", help="Set to 'True' to run the tests on a subset of images.")
    parser.add_argument("-l", "--load_data", type=str, default="True", 
                        help="Set to 'True' to load bootstrapped numpy files from file server if they exist, else calculate them.")
    args = parser.parse_args()

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler("statistical_tests_by_category.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Convert string inputs to booleans
    TEST = args.test.lower() == "true"
    LOAD_DATA = args.load_data.lower() == "true"

    # CSV-Datei mit Maskenkategorien einlesen
    df_mask = pd.read_csv("mask_sizes_with_category.csv")

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

    # Get all file names that are in all directories
    image_files = list(set(real_files) & set(lama_files) & set(opencv_files) & set(sd_files))
    logger.info(f"Found {len(image_files)} images in all directories")

    # Define a fixed image size
    target_size = (256, 256)  # Resize all images to this size

    # Load the images
    logger.info("Loading the images")
    real_images = list(image_generator(images_dir, image_files))
    lama_images = list(image_generator(results_dir_lama, image_files))
    opencv_images = list(image_generator(results_dir_opencv, image_files))
    sd_images = list(image_generator(results_dir_sd, image_files))

    # If TEST is True, only use the first 100 images for testing
    if TEST:
        real_images = real_images[:100]
        lama_images = lama_images[:100]
        opencv_images = opencv_images[:100]
        sd_images = sd_images[:100]
        N_BOOTSTRAPS = 5
    else:
        N_BOOTSTRAPS = 500

    logger.info(f"Loaded {len(real_images)} Real images")
    logger.info(f"Loaded {len(lama_images)} Lama images")
    logger.info(f"Loaded {len(opencv_images)} OpenCV images")
    logger.info(f"Loaded {len(sd_images)} SD images")

    # Preprocess the images (resize and convert to tensors)
    real_images_tensor = preprocess_images(real_images, target_size)
    lama_images_tensor = preprocess_images(lama_images, target_size)
    opencv_images_tensor = preprocess_images(opencv_images, target_size)
    sd_images_tensor = preprocess_images(sd_images, target_size)

    # Bilder nach Kategorien filtern
    real_images_tensor_small = get_mask_array(real_images_tensor, df_mask, "klein", image_files)
    real_images_tensor_middle = get_mask_array(real_images_tensor, df_mask, "mittel", image_files)
    real_images_tensor_big = get_mask_array(real_images_tensor, df_mask, "groß", image_files)

    lama_images_tensor_small = get_mask_array(lama_images_tensor, df_mask, "klein", image_files)
    lama_images_tensor_middle = get_mask_array(lama_images_tensor, df_mask, "mittel", image_files)
    lama_images_tensor_big = get_mask_array(lama_images_tensor, df_mask, "groß", image_files)

    opencv_images_tensor_small = get_mask_array(opencv_images_tensor, df_mask, "klein", image_files)
    opencv_images_tensor_middle = get_mask_array(opencv_images_tensor, df_mask, "mittel", image_files)
    opencv_images_tensor_big = get_mask_array(opencv_images_tensor, df_mask, "groß", image_files)

    sd_images_tensor_small = get_mask_array(sd_images_tensor, df_mask, "klein", image_files)
    sd_images_tensor_middle = get_mask_array(sd_images_tensor, df_mask, "mittel", image_files)
    sd_images_tensor_big = get_mask_array(sd_images_tensor, df_mask, "groß", image_files)

    # Calculate the FID score between the real and generated images for each category
    fid_score_lama_small = calculate_fid(real_images_tensor_small, lama_images_tensor_small)
    fid_score_lama_middle = calculate_fid(real_images_tensor_middle, lama_images_tensor_middle)
    fid_score_lama_big = calculate_fid(real_images_tensor_big, lama_images_tensor_big)

    fid_score_opencv_small = calculate_fid(real_images_tensor_small, opencv_images_tensor_small)
    fid_score_opencv_middle = calculate_fid(real_images_tensor_middle, opencv_images_tensor_middle)
    fid_score_opencv_big = calculate_fid(real_images_tensor_big, opencv_images_tensor_big)

    fid_score_sd_small = calculate_fid(real_images_tensor_small, sd_images_tensor_small)
    fid_score_sd_middle = calculate_fid(real_images_tensor_middle, sd_images_tensor_middle)
    fid_score_sd_big = calculate_fid(real_images_tensor_big, sd_images_tensor_big)

    # Store the FID scores in a dictionary and save the dictionary to a file
    fid_scores = {
        "lama_small": fid_score_lama_small,
        "lama_middle": fid_score_lama_middle,
        "lama_big": fid_score_lama_big,
        "opencv_small": fid_score_opencv_small,
        "opencv_middle": fid_score_opencv_middle,
        "opencv_big": fid_score_opencv_big,
        "sd_small": fid_score_sd_small,
        "sd_middle": fid_score_sd_middle,
        "sd_big": fid_score_sd_big,
    }

    with open("fid_by_category.json", "w") as f:
        json.dump(fid_scores, f)

    # Calculate the FID score using bootstrapping for each category and method.
    # If LOAD_DATA is True and the file exists, load it; otherwise, compute and save.
    
    # Helper function for bootstrapping
    def load_or_compute_bootstrap(file_name, real_tensor, gen_tensor, n_bootstraps):
        if LOAD_DATA and os.path.exists(file_name):
            bs = np.load(file_name)
            logger.info(f"Loaded bootstrapped FID from {file_name}")
        else:
            bs = bootstrap_fid(real_tensor, gen_tensor, n_bootstraps)
            np.save(file_name, bs)
            logger.info(f"Computed and saved bootstrapped FID to {file_name}")
        logger.info(f"Mean FID from {file_name}: {np.mean(bs)}, std: {np.std(bs)}")
        return bs

    # Lama bootstraps
    fid_score_lama_small_bootstrap = load_or_compute_bootstrap("fid_lama_small.npy", real_images_tensor_small, lama_images_tensor_small, N_BOOTSTRAPS)
    fid_score_lama_middle_bootstrap = load_or_compute_bootstrap("fid_lama_middle.npy", real_images_tensor_middle, lama_images_tensor_middle, N_BOOTSTRAPS)
    fid_score_lama_big_bootstrap = load_or_compute_bootstrap("fid_lama_big.npy", real_images_tensor_big, lama_images_tensor_big, N_BOOTSTRAPS)

    # OpenCV bootstraps
    fid_score_opencv_small_bootstrap = load_or_compute_bootstrap("fid_opencv_small.npy", real_images_tensor_small, opencv_images_tensor_small, N_BOOTSTRAPS)
    fid_score_opencv_middle_bootstrap = load_or_compute_bootstrap("fid_opencv_middle.npy", real_images_tensor_middle, opencv_images_tensor_middle, N_BOOTSTRAPS)
    fid_score_opencv_big_bootstrap = load_or_compute_bootstrap("fid_opencv_big.npy", real_images_tensor_big, opencv_images_tensor_big, N_BOOTSTRAPS)

    # SD bootstraps
    fid_score_sd_small_bootstrap = load_or_compute_bootstrap("fid_sd_small.npy", real_images_tensor_small, sd_images_tensor_small, N_BOOTSTRAPS)
    fid_score_sd_middle_bootstrap = load_or_compute_bootstrap("fid_sd_middle.npy", real_images_tensor_middle, sd_images_tensor_middle, N_BOOTSTRAPS)
    fid_score_sd_big_bootstrap = load_or_compute_bootstrap("fid_sd_big.npy", real_images_tensor_big, sd_images_tensor_big, N_BOOTSTRAPS)

    # Check normality and perform statistical tests (the code remains the same)
    lama_small_normality, lama_small_normality_details = check_normality(fid_score_lama_small_bootstrap)
    logger.info(f"Normality test for FID (Lama) Small Mask: {lama_small_normality},details,{lama_small_normality_details}")

    lama_middle_normality, lama_middle_normality_details = check_normality(fid_score_lama_middle_bootstrap)
    logger.info(f"Normality test for FID (Lama) Middle Mask: {lama_middle_normality}, details: {lama_middle_normality_details}")

    lama_big_normality, lama_big_normality_details = check_normality(fid_score_lama_big_bootstrap)
    logger.info(f"Normality test for FID (Lama) Big Mask: {lama_big_normality}, details: {lama_big_normality_details}")

    opencv_small_normality, opencv_small_normality_details = check_normality(fid_score_opencv_small_bootstrap)
    logger.info(f"Normality test for FID (OpenCV) Small Mask: {opencv_small_normality}, details{opencv_small_normality_details}")

    opencv_middle_normality, opencv_middle_normality_details = check_normality(fid_score_opencv_middle_bootstrap)
    logger.info(f"Normality test for FID (OpenCV) Middle Mask: {opencv_middle_normality}, details: {opencv_middle_normality_details}")

    opencv_big_normality, opencv_big_normality_details = check_normality(fid_score_opencv_big_bootstrap)
    logger.info(f"Normality test for FID (OpenCV) Big Mask: {opencv_big_normality}, details: {opencv_big_normality_details}")

    sd_small_normality, sd_small_normality_details = check_normality(fid_score_sd_small_bootstrap)
    logger.info(f"Normality test for FID (SD) Small Mask: {sd_small_normality}, details: {sd_small_normality_details}")

    sd_middle_normality, sd_middle_normality_details = check_normality(fid_score_sd_middle_bootstrap)
    logger.info(f"Normality test for FID (SD) Middle Mask: {sd_middle_normality}, details: {sd_middle_normality_details}")

    sd_big_normality, sd_big_normality_details = check_normality(fid_score_sd_big_bootstrap)
    logger.info(f"Normality test for FID (SD) Big Mask: {sd_big_normality},
    details: {sd_big_normality_details}")

    # Perform statistical tests

    # Wilcoxon signed-rank test between OpenCV and Lama for each mask category
    p_value_opencv_lama_small_ws, obs_diff_opencv_lama_small_ws = wilcoxon_signed_rank_test(fid_score_opencv_small_bootstrap, fid_score_lama_small_bootstrap)
    logger.info(f"Wilcoxon (OpenCV vs Lama Small Mask): p-value = {p_value_opencv_lama_small_ws}, observed diff = {obs_diff_opencv_lama_small_ws}")

    p_value_opencv_lama_middle_ws, obs_diff_opencv_lama_middle_ws = wilcoxon_signed_rank_test(fid_score_opencv_middle_bootstrap, fid_score_lama_middle_bootstrap)
    logger.info(f"Wilcoxon (OpenCV vs Lama Middle Mask): p-value = {p_value_opencv_lama_middle_ws}, observed diff = {obs_diff_opencv_lama_middle_ws}")

    p_value_opencv_lama_big_ws, obs_diff_opencv_lama_big_ws = wilcoxon_signed_rank_test(fid_score_opencv_big_bootstrap, fid_score_lama_big_bootstrap)
    logger.info(f"Wilcoxon (OpenCV vs Lama Big Mask): p-value = {p_value_opencv_lama_big_ws}, observed diff = {obs_diff_opencv_lama_big_ws}")

    # Two-sided t-test between OpenCV and Lama for each mask category
    p_value_opencv_lama_small_tt, obs_diff_opencv_lama_small_tt = two_sided_t_test(fid_score_opencv_small_bootstrap, fid_score_lama_small_bootstrap)
    logger.info(f"Two-sided t-test (OpenCV vs Lama Small Mask): p-value = {p_value_opencv_lama_small_tt}, observed diff = {obs_diff_opencv_lama_small_tt}")

    p_value_opencv_lama_middle_tt, obs_diff_opencv_lama_middle_tt = two_sided_t_test(fid_score_opencv_middle_bootstrap, fid_score_lama_middle_bootstrap)
    logger.info(f"Two-sided t-test (OpenCV vs Lama Middle Mask): p-value = {p_value_opencv_lama_middle_tt}, observed diff = {obs_diff_opencv_lama_middle_tt}")

    p_value_opencv_lama_big_tt, obs_diff_opencv_lama_big_tt = two_sided_t_test(fid_score_opencv_big_bootstrap, fid_score_lama_big_bootstrap)
    logger.info(f"Two-sided t-test (OpenCV vs Lama Big Mask): p-value = {p_value_opencv_lama_big_tt}, observed diff = {obs_diff_opencv_lama_big_tt}")

    # Wilcoxon signed-rank test between SD and Lama for each mask category
    p_value_sd_lama_small_ws, obs_diff_sd_lama_small_ws = wilcoxon_signed_rank_test(fid_score_sd_small_bootstrap, fid_score_lama_small_bootstrap)
    logger.info(f"Wilcoxon (SD vs Lama Small Mask): p-value = {p_value_sd_lama_small_ws}, observed diff = {obs_diff_sd_lama_small_ws}")

    p_value_sd_lama_middle_ws, obs_diff_sd_lama_middle_ws = wilcoxon_signed_rank_test(fid_score_sd_middle_bootstrap, fid_score_lama_middle_bootstrap)
    logger.info(f"Wilcoxon (SD vs Lama Middle Mask): p-value = {p_value_sd_lama_middle_ws}, observed diff = {obs_diff_sd_lama_middle_ws}")

    p_value_sd_lama_big_ws, obs_diff_sd_lama_big_ws = wilcoxon_signed_rank_test(fid_score_sd_big_bootstrap, fid_score_lama_big_bootstrap)
    logger.info(f"Wilcoxon (SD vs Lama Big Mask): p-value = {p_value_sd_lama_big_ws}, observed diff = {obs_diff_sd_lama_big_ws}")

    # Two-sided t-test between SD and Lama for each mask category
    p_value_sd_lama_small_tt, obs_diff_sd_lama_small_tt = two_sided_t_test(fid_score_sd_small_bootstrap, fid_score_lama_small_bootstrap)
    logger.info(f"Two-sided t-test (SD vs Lama Small Mask): p-value = {p_value_sd_lama_small_tt}, observed diff = {obs_diff_sd_lama_small_tt}")

    p_value_sd_lama_middle_tt, obs_diff_sd_lama_middle_tt = two_sided_t_test(fid_score_sd_middle_bootstrap, fid_score_lama_middle_bootstrap)
    logger.info(f"Two-sided t-test (SD vs Lama Middle Mask): p-value = {p_value_sd_lama_middle_tt}, observed diff = {obs_diff_sd_lama_middle_tt}")

    p_value_sd_lama_big_tt, obs_diff_sd_lama_big_tt = two_sided_t_test(fid_score_sd_big_bootstrap, fid_score_lama_big_bootstrap)
    logger.info(f"Two-sided t-test (SD vs Lama Big Mask): p-value = {p_value_sd_lama_big_tt}, observed diff = {obs_diff_sd_lama_big_tt}")

    # Wilcoxon signed-rank test between SD and OpenCV for each mask category
    p_value_sd_opencv_small_ws, obs_diff_sd_opencv_small_ws = wilcoxon_signed_rank_test(fid_score_sd_small_bootstrap, fid_score_opencv_small_bootstrap)
    logger.info(f"Wilcoxon (SD vs OpenCV Small Mask): p-value = {p_value_sd_opencv_small_ws}, observed diff = {obs_diff_sd_opencv_small_ws}")

    p_value_sd_opencv_middle_ws, obs_diff_sd_opencv_middle_ws = wilcoxon_signed_rank_test(fid_score_sd_middle_bootstrap, fid_score_opencv_middle_bootstrap)
    logger.info(f"Wilcoxon (SD vs OpenCV Middle Mask): p-value = {p_value_sd_opencv_middle_ws}, observed diff = {obs_diff_sd_opencv_middle_ws}")

    p_value_sd_opencv_big_ws, obs_diff_sd_opencv_big_ws = wilcoxon_signed_rank_test(fid_score_sd_big_bootstrap, fid_score_opencv_big_bootstrap)
    logger.info(f"Wilcoxon (SD vs OpenCV Big Mask): p-value = {p_value_sd_opencv_big_ws}, observed diff = {obs_diff_sd_opencv_big_ws}")

    # Two-sided t-test between SD and OpenCV for each mask category
    p_value_sd_opencv_small_tt, obs_diff_sd_opencv_small_tt = two_sided_t_test(fid_score_sd_small_bootstrap, fid_score_opencv_small_bootstrap)
    logger.info(f"Two-sided t-test (SD vs OpenCV Small Mask): p-value = {p_value_sd_opencv_small_tt}, observed diff = {obs_diff_sd_opencv_small_tt}")

    p_value_sd_opencv_middle_tt, obs_diff_sd_opencv_middle_tt = two_sided_t_test(fid_score_sd_middle_bootstrap, fid_score_opencv_middle_bootstrap)
    logger.info(f"Two-sided t-test (SD vs OpenCV Middle Mask): p-value = {p_value_sd_opencv_middle_tt}, observed diff = {obs_diff_sd_opencv_middle_tt}")

    p_value_sd_opencv_big_tt, obs_diff_sd_opencv_big_tt = two_sided_t_test(fid_score_sd_big_bootstrap, fid_score_opencv_big_bootstrap)
    logger.info(f"Two-sided t-test (SD vs OpenCV Big Mask): p-value = {p_value_sd_opencv_big_tt}, observed diff = {obs_diff_sd_opencv_big_tt}")

    # Save the results of the statistical tests to a file
    results = {
        "p_value_opencv_lama_small_ws": p_value_opencv_lama_small_ws,
        "obs_diff_opencv_lama_small_ws": obs_diff_opencv_lama_small_ws,
        "p_value_opencv_lama_middle_ws": p_value_opencv_lama_middle_ws,
        "obs_diff_opencv_lama_middle_ws": obs_diff_opencv_lama_middle_ws,
        "p_value_opencv_lama_big_ws": p_value_opencv_lama_big_ws,
        "obs_diff_opencv_lama_big_ws": obs_diff_opencv_lama_big_ws,
        "p_value_opencv_lama_small_tt": p_value_opencv_lama_small_tt,
        "obs_diff_opencv_lama_small_tt": obs_diff_opencv_lama_small_tt,
        "p_value_opencv_lama_middle_tt": p_value_opencv_lama_middle_tt,
        "obs_diff_opencv_lama_middle_tt": obs_diff_opencv_lama_middle_tt,
        "p_value_opencv_lama_big_tt": p_value_opencv_lama_big_tt,
        "obs_diff_opencv_lama_big_tt": obs_diff_opencv_lama_big_tt,
        "p_value_sd_lama_small_ws": p_value_sd_lama_small_ws,
        "obs_diff_sd_lama_small_ws": obs_diff_sd_lama_small_ws,
        "p_value_sd_lama_middle_ws": p_value_sd_lama_middle_ws,
        "obs_diff_sd_lama_middle_ws": obs_diff_sd_lama_middle_ws,
        "p_value_sd_lama_big_ws": p_value_sd_lama_big_ws,
        "obs_diff_sd_lama_big_ws": obs_diff_sd_lama_big_ws,
        "p_value_sd_lama_small_tt": p_value_sd_lama_small_tt,
        "obs_diff_sd_lama_small_tt": obs_diff_sd_lama_small_tt,
        "p_value_sd_lama_middle_tt": p_value_sd_lama_middle_tt,
        "obs_diff_sd_lama_middle_tt": obs_diff_sd_lama_middle_tt,
        "p_value_sd_lama_big_tt": p_value_sd_lama_big_tt,
        "obs_diff_sd_lama_big_tt": obs_diff_sd_lama_big_tt,
        "p_value_sd_opencv_small_ws": p_value_sd_opencv_small_ws,
        "obs_diff_sd_opencv_small_ws": obs_diff_sd_opencv_small_ws,
        "p_value_sd_opencv_middle_ws": p_value_sd_opencv_middle_ws,
        "obs_diff_sd_opencv_middle_ws": obs_diff_sd_opencv_middle_ws,
        "p_value_sd_opencv_big_ws": p_value_sd_opencv_big_ws,
        "obs_diff_sd_opencv_big_ws": obs_diff_sd_opencv_big_ws,
        "p_value_sd_opencv_small_tt": p_value_sd_opencv_small_tt,
        "obs_diff_sd_opencv_small_tt": obs_diff_sd_opencv_small_tt,
        "p_value_sd_opencv_middle_tt": p_value_sd_opencv_middle_tt,
        "obs_diff_sd_opencv_middle_tt": obs_diff_sd_opencv_middle_tt,
        "p_value_sd_opencv_big_tt": p_value_sd_opencv_big_tt,
        "obs_diff_sd_opencv_big_tt": obs_diff_sd_opencv_big_tt,
    }

    with open("statistical_tests_by_category.json", "w") as f:
        json.dump(results, f)
