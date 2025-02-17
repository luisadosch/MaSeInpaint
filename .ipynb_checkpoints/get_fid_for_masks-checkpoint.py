import pandas as pd
import json
import os
import torch
from statistical_tests import image_generator, preprocess_images, calculate_fid, images_dir, masks_dir, results_dir_lama, results_dir_opencv, results_dir_sd, real_files, lama_files, opencv_files, sd_files, image_files, real_images, lama_images, opencv_images, sd_images,
real_images_tensor, lama_images_tensor, opencv_images_tensor , sd_images_tensor
import logging

# Logger einrichten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSV-Datei mit Maskenkategorien einlesen
df_mask = pd.read_csv("mask_sizes_with_category.csv")



logger.info(f"Found {len(image_files)} images in all directories")


logger.info(f"Loaded {len(real_images)} Real images")
logger.info(f"Loaded {len(lama_images)} Lama images")
logger.info(f"Loaded {len(opencv_images)} OpenCV images")
logger.info(f"Loaded {len(sd_images)} SD images")



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
    logger.info(f"Loaded {category} category")
    # Dateinamen der Bilder in der Kategorie
    category_files = df_mask[df_mask["mask_category"] == category]["Bildname"].tolist()
    logger.debug(f"Filtered file names: {category_files}")
    
    # Indizes der Bilder in der Kategorie
    indices = [i for i, fname in enumerate(image_files) if fname in category_files]
    logger.debug(f"Found indices: {indices}")

    # Falls keine Bilder gefunden wurden
    if not indices:
        logger.warning("No images found for this category!")

    # Wähle die entsprechenden Bilder aus
    result = images_tensor[indices]
    logger.info(f"Filtered tensor shape: {result.shape}")

    
    return result

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

logger.info("FID scores saved to fid_scores.json")

