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
    args = parser.parse_args()

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("fid_per_catergory.log")
    file_handler.setLevel(logging.INFO)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


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

    # Get all file names that are in all three directories
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


    # Convert string input to boolean
    TEST = args.test.lower() == "true"

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

    #filter 


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

    # Calculate the FID score between the real and generated images
    fid_score_lama_small = calculate_fid(real_images_tensor_small, lama_images_tensor_small)
    fid_score_lama_middle = calculate_fid(real_images_tensor_middle, lama_images_tensor_middle)
    fid_score_lama_big = calculate_fid(real_images_tensor_big, lama_images_tensor_big)

    fid_score_opencv_small = calculate_fid(real_images_tensor_small, opencv_images_tensor_small)
    fid_score_opencv_middle = calculate_fid(real_images_tensor_middle, opencv_images_tensor_middle)
    fid_score_opencv_big = calculate_fid(real_images_tensor_big, opencv_images_tensor_big)

    fid_score_sd_small = calculate_fid(real_images_tensor_small, sd_images_tensor_small)
    fid_score_sd_middle = calculate_fid(real_images_tensor_middle, sd_images_tensor_middle)
    fid_score_sd_big = calculate_fid(real_images_tensor_big, sd_images_tensor_big)

    logger.info(f"FID (Lama) Small Mask: {fid_score_lama_small}")
    logger.info(f"FID (Lama) Middle Mask: {fid_score_lama_middle}")
    logger.info(f"FID (Lama) Big Mask: {fid_score_lama_big}")

    logger.info(f"FID (OpenCV) Small Mask: {fid_score_opencv_small}")
    logger.info(f"FID (OpenCV) Middle Mask: {fid_score_opencv_middle}")
    logger.info(f"FID (OpenCV) Big Mask: {fid_score_opencv_big}")

    logger.info(f"FID (SD) Small Mask: {fid_score_sd_small}")
    logger.info(f"FID (SD) Middle Mask: {fid_score_sd_middle}")
    logger.info(f"FID (SD) Big Mask: {fid_score_sd_big}")

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

    with open("get_fid_by_category.json", "w") as f:
        json.dump(fid_scores, f)

    # Calculate the FID score using bootstrapping
    fid_score_lama_small_bootstrap = bootstrap_fid(real_images_tensor_small, lama_images_tensor_small, N_BOOTSTRAPS)
    logger.info(f"Bootstrap FID (Lama) Small Mask: {fid_score_lama_small_bootstrap}")
    # Save the bootstrapped FID score to a numpy file
    np.save("fid_lama_small.npy", fid_score_lama_small_bootstrap)
    logger.info("Saved bootstrapped FID (Lama) Small Mask to fid_lama_small.npy")
    #Check normality
    lama_small_normality, lama_small_normality_details = check_normality(fid_score_lama_small_bootstrap)
    logger.info(f"Normality test for FID (Lama) Small Mask: {lama_small_normality}")


    fid_score_lama_middle_bootstrap = bootstrap_fid(real_images_tensor_middle, lama_images_tensor_middle, N_BOOTSTRAPS)
    logger.info(f"Bootstrap FID (Lama) Middle Mask: {fid_score_lama_middle_bootstrap}")
    # Save the bootstrapped FID score to a numpy file
    np.save("fid_lama_middle.npy", fid_score_lama_middle_bootstrap)
    logger.info("Saved bootstrapped FID (Lama) Middle Mask to fid_lama_middle.npy")
    #Check normality
    lama_middle_normality, lama_middle_normality_details = check_normality(fid_score_lama_middle_bootstrap)
    logger.info(f"Normality test for FID (Lama) Middle Mask: {lama_middle_normality}")


    fid_score_lama_big_bootstrap = bootstrap_fid(real_images_tensor_big, lama_images_tensor_big, N_BOOTSTRAPS)
    logger.info(f"Bootstrap FID (Lama) Big Mask: {fid_score_lama_big_bootstrap}")
    # Save the bootstrapped FID score to a numpy file
    np.save("fid_lama_big.npy", fid_score_lama_big_bootstrap)
    logger.info("Saved bootstrapped FID (Lama) Big Mask to fid_lama_big.npy")
    #Check normality
    lama_big_normality, lama_big_normality_details = check_normality(fid_score_lama_big_bootstrap)
    logger.info(f"Normality test for FID (Lama) Big Mask: {lama_big_normality}")

    fid_score_opencv_small_bootstrap = bootstrap_fid(real_images_tensor_small, opencv_images_tensor_small, N_BOOTSTRAPS)
    logger.info(f"Bootstrap FID (OpenCV) Small Mask: {fid_score_opencv_small_bootstrap}")
    # Save the bootstrapped FID score to a numpy file
    np.save("fid_opencv_small.npy", fid_score_opencv_small_bootstrap)
    logger.info("Saved bootstrapped FID (OpenCV) Small Mask to fid_opencv_small.npy")
    #Check normality
    opencv_small_normality, opencv_small_normality_details = check_normality(fid_score_opencv_small_bootstrap)
    logger.info(f"Normality test for FID (OpenCV) Small Mask: {opencv_small_normality}")


    fid_score_opencv_middle_bootstrap = bootstrap_fid(real_images_tensor_middle, opencv_images_tensor_middle, N_BOOTSTRAPS)
    logger.info(f"Bootstrap FID (OpenCV) Middle Mask: {fid_score_opencv_middle_bootstrap}")
    # Save the bootstrapped FID score to a numpy file
    np.save("fid_opencv_middle.npy", fid_score_opencv_middle_bootstrap)
    logger.info("Saved bootstrapped FID (OpenCV) Middle Mask to fid_opencv_middle.npy")
    #Check normality
    opencv_middle_normality, opencv_middle_normality_details = check_normality(fid_score_opencv_middle_bootstrap)
    logger.info(f"Normality test for FID (OpenCV) Middle Mask: {opencv_middle_normality}")


    fid_score_opencv_big_bootstrap = bootstrap_fid(real_images_tensor_big, opencv_images_tensor_big, N_BOOTSTRAPS)
    logger.info(f"Bootstrap FID (OpenCV) Big Mask: {fid_score_opencv_big_bootstrap}")
    # Save the bootstrapped FID score to a numpy file
    np.save("fid_opencv_big.npy", fid_score_opencv_big_bootstrap)
    logger.info("Saved bootstrapped FID (OpenCV) Big Mask to fid_opencv_big.npy")
    #Check normality
    opencv_big_normality, opencv_big_normality_details = check_normality(fid_score_opencv_big_bootstrap)
    logger.info(f"Normality test for FID (OpenCV) Big Mask: {opencv_big_normality}")


    fid_score_sd_small_bootstrap = bootstrap_fid(real_images_tensor_small, sd_images_tensor_small, N_BOOTSTRAPS)
    logger.info(f"Bootstrap FID (SD) Small Mask: {fid_score_sd_small_bootstrap}")
    # Save the bootstrapped FID score to a numpy file
    np.save("fid_sd_small.npy", fid_score_sd_small_bootstrap)
    logger.info("Saved bootstrapped FID (SD) Small Mask to fid_sd_small.npy")
    #Check normality
    sd_small_normality, sd_small_normality_details = check_normality(fid_score_sd_small_bootstrap)
    logger.info(f"Normality test for FID (SD) Small Mask: {sd_small_normality}")

    fid_score_sd_middle_bootstrap = bootstrap_fid(real_images_tensor_middle, sd_images_tensor_middle, N_BOOTSTRAPS)
    logger.info(f"Bootstrap FID (SD) Middle Mask: {fid_score_sd_middle_bootstrap}")
    # Save the bootstrapped FID score to a numpy file
    np.save("fid_sd_middle.npy", fid_score_sd_middle_bootstrap)
    logger.info("Saved bootstrapped FID (SD) Middle Mask to fid_sd_middle.npy")
    #Check normality
    sd_middle_normality, sd_middle_normality_details = check_normality(fid_score_sd_middle_bootstrap)
    logger.info(f"Normality test for FID (SD) Middle Mask: {sd_middle_normality}")

    fid_score_sd_big_bootstrap = bootstrap_fid(real_images_tensor_big, sd_images_tensor_big, N_BOOTSTRAPS)
    logger.info(f"Bootstrap FID (SD) Big Mask: {fid_score_sd_big_bootstrap}")
    # Save the bootstrapped FID score to a numpy file
    np.save("fid_sd_big.npy", fid_score_sd_big_bootstrap)
    logger.info("Saved bootstrapped FID (SD) Big Mask to fid_sd_big.npy")
    #Check normality
    sd_big_normality, sd_big_normality_details = check_normality(fid_score_sd_big_bootstrap)
    logger.info(f"Normality test for FID (SD) Big Mask: {sd_big_normality}")


    # Perform statistical tests
    logger.info("Performing statistical tests")

    # Wilcoxon signed-rank test
    
    # Perform Wilcoxon signed-rank test between OpenCV and Lama for each mask category

    #Small Mask
    p_value_opencv_lama_small_ws, obs_diff_opencv_lama_small_ws = wilcoxon_signed_rank_test(fid_score_opencv_small_bootstrap, fid_score_lama_small_bootstrap)
    logger.info(f"Wilcoxon signed-rank test between OpenCV and Lama Small Mask: p-value = {p_value_opencv_lama_small_ws}, observed difference = {obs_diff_opencv_lama_small_ws}")

    #Middle Mask
    p_value_opencv_lama_middle_ws, obs_diff_opencv_lama_middle_ws = wilcoxon_signed_rank_test(fid_score_opencv_middle_bootstrap, fid_score_lama_middle_bootstrap)
    logger.info(f"Wilcoxon signed-rank test between OpenCV and Lama Middle Mask: p-value = {p_value_opencv_lama_middle_ws}, observed difference = {obs_diff_opencv_lama_middle_ws}")

    #Big Mask
    p_value_opencv_lama_big_ws, obs_diff_opencv_lama_big_ws = wilcoxon_signed_rank_test(fid_score_opencv_big_bootstrap, fid_score_lama_big_bootstrap)
    logger.info(f"Wilcoxon signed-rank test between OpenCV and Lama Big Mask: p-value = {p_value_opencv_lama_big_ws}, observed difference = {obs_diff_opencv_lama_big_ws}")

    # Perform two-sided t-test between OpenCV and Lama for each mask category

    #Small Mask
    p_value_opencv_lama_small_tt, obs_diff_opencv_lama_small_tt = two_sided_t_test(fid_score_opencv_small_bootstrap, fid_score_lama_small_bootstrap)
    logger.info(f"Two-sided t-test between OpenCV and Lama Small Mask: p-value = {p_value_opencv_lama_small_tt}, observed difference = {obs_diff_opencv_lama_small_tt}")

    #Middle Mask
    p_value_opencv_lama_middle_tt, obs_diff_opencv_lama_middle_tt = two_sided_t_test(fid_score_opencv_middle_bootstrap, fid_score_lama_middle_bootstrap)
    logger.info(f"Two-sided t-test between OpenCV and Lama Middle Mask: p-value = {p_value_opencv_lama_middle_tt}, observed difference = {obs_diff_opencv_lama_middle_tt}")

    #Big Mask
    p_value_opencv_lama_big_tt, obs_diff_opencv_lama_big_tt = two_sided_t_test(fid_score_opencv_big_bootstrap, fid_score_lama_big_bootstrap)
    logger.info(f"Two-sided t-test between OpenCV and Lama Big Mask: p-value = {p_value_opencv_lama_big_tt}, observed difference = {obs_diff_opencv_lama_big_tt}")

    # Perform Wilcoxon signed-rank test between SD and Lama for each mask category

    #Small Mask
    p_value_sd_lama_small_ws, obs_diff_sd_lama_small_ws = wilcoxon_signed_rank_test(fid_score_sd_small_bootstrap, fid_score_lama_small_bootstrap)
    logger.info(f"Wilcoxon signed-rank test between SD and Lama Small Mask: p-value = {p_value_sd_lama_small_ws}, observed difference = {obs_diff_sd_lama_small_ws}")

    #Middle Mask
    p_value_sd_lama_middle_ws, obs_diff_sd_lama_middle_ws = wilcoxon_signed_rank_test(fid_score_sd_middle_bootstrap, fid_score_lama_middle_bootstrap)
    logger.info(f"Wilcoxon signed-rank test between SD and Lama Middle Mask: p-value = {p_value_sd_lama_middle_ws}, observed difference = {obs_diff_sd_lama_middle_ws}")

    #Big Mask
    p_value_sd_lama_big_ws, obs_diff_sd_lama_big_ws = wilcoxon_signed_rank_test(fid_score_sd_big_bootstrap, fid_score_lama_big_bootstrap)
    logger.info(f"Wilcoxon signed-rank test between SD and Lama Big Mask: p-value = {p_value_sd_lama_big_ws}, observed difference = {obs_diff_sd_lama_big_ws}")

    # Perform two-sided t-test between SD and Lama for each mask category

    #Small Mask
    p_value_sd_lama_small_tt, obs_diff_sd_lama_small_tt = two_sided_t_test(fid_score_sd_small_bootstrap, fid_score_lama_small_bootstrap)
    logger.info(f"Two-sided t-test between SD and Lama Small Mask: p-value = {p_value_sd_lama_small_tt}, observed difference = {obs_diff_sd_lama_small_tt}")

    #Middle Mask
    p_value_sd_lama_middle_tt, obs_diff_sd_lama_middle_tt = two_sided_t_test(fid_score_sd_middle_bootstrap, fid_score_lama_middle_bootstrap)
    logger.info(f"Two-sided t-test between SD and Lama Middle Mask: p-value = {p_value_sd_lama_middle_tt}, observed difference = {obs_diff_sd_lama_middle_tt}")

    #Big Mask
    p_value_sd_lama_big_tt, obs_diff_sd_lama_big_tt = two_sided_t_test(fid_score_sd_big_bootstrap, fid_score_lama_big_bootstrap)
    logger.info(f"Two-sided t-test between SD and Lama Big Mask: p-value = {p_value_sd_lama_big_tt}, observed difference = {obs_diff_sd_lama_big_tt}")

    # Perform Wilcoxon signed-rank test between SD and OpenCV for each mask category
    
    #Small Mask
    p_value_sd_opencv_small_ws, obs_diff_sd_opencv_small_ws = wilcoxon_signed_rank_test(fid_score_sd_small_bootstrap, fid_score_opencv_small_bootstrap)
    logger.info(f"Wilcoxon signed-rank test between SD and OpenCV Small Mask: p-value = {p_value_sd_opencv_small_ws}, observed difference = {obs_diff_sd_opencv_small_ws}")

    #Middle Mask
    p_value_sd_opencv_middle_ws, obs_diff_sd_opencv_middle_ws = wilcoxon_signed_rank_test(fid_score_sd_middle_bootstrap, fid_score_opencv_middle_bootstrap)
    logger.info(f"Wilcoxon signed-rank test between SD and OpenCV Middle Mask: p-value = {p_value_sd_opencv_middle_ws}, observed difference = {obs_diff_sd_opencv_middle_ws}")

    #Big Mask
    p_value_sd_opencv_big_ws, obs_diff_sd_opencv_big_ws = wilcoxon_signed_rank_test(fid_score_sd_big_bootstrap, fid_score_opencv_big_bootstrap)
    logger.info(f"Wilcoxon signed-rank test between SD and OpenCV Big Mask: p-value = {p_value_sd_opencv_big_ws}, observed difference = {obs_diff_sd_opencv_big_ws}")

    # Perform two-sided t-test between SD and OpenCV for each mask category

    #Small Mask
    p_value_sd_opencv_small_tt, obs_diff_sd_opencv_small_tt = two_sided_t_test(fid_score_sd_small_bootstrap, fid_score_opencv_small_bootstrap)
    logger.info(f"Two-sided t-test between SD and OpenCV Small Mask: p-value = {p_value_sd_opencv_small_tt}, observed difference = {obs_diff_sd_opencv_small_tt}")

    #Middle Mask
    p_value_sd_opencv_middle_tt, obs_diff_sd_opencv_middle_tt = two_sided_t_test(fid_score_sd_middle_bootstrap, fid_score_opencv_middle_bootstrap)
    logger.info(f"Two-sided t-test between SD and OpenCV Middle Mask: p-value = {p_value_sd_opencv_middle_tt}, observed difference = {obs_diff_sd_opencv_middle_tt}")

    #Big Mask
    p_value_sd_opencv_big_tt, obs_diff_sd_opencv_big_tt = two_sided_t_test(fid_score_sd_big_bootstrap, fid_score_opencv_big_bootstrap)
    logger.info(f"Two-sided t-test between SD and OpenCV Big Mask: p-value = {p_value_sd_opencv_big_tt}, observed difference = {obs_diff_sd_opencv_big_tt}")

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


    






        
