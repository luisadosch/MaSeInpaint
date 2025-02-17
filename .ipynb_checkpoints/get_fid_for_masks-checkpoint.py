import pandas as pd
import json
import os
import torch
from statistical_tests import image_generator, preprocess_images, calculate_fid
import logging

# Logger einrichten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

logger.info(f"Loaded {len(real_images)} Real images")
logger.info(f"Loaded {len(lama_images)} Lama images")
logger.info(f"Loaded {len(opencv_images)} OpenCV images")
logger.info(f"Loaded {len(sd_images)} SD images")

# Preprocess the images (resize and convert to tensors)
real_images_tensor = preprocess_images(real_images, target_size)
lama_images_tensor = preprocess_images(lama_images, target_size)
opencv_images_tensor = preprocess_images(opencv_images, target_size)
sd_images_tensor = preprocess_images(sd_images, target_size)

def get_mask_array(images_tensor, df_mask, category):
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
    indices = [i for i, fname in enumerate(image_files) if fname in category_files]
    
    # Bilder in der Kategorie auswählen
    return images_tensor[indices]

# Bilder nach Kategorien filtern
real_images_tensor_small = get_mask_array(real_images_tensor, df_mask, "klein")
real_images_tensor_middle = get_mask_array(real_images_tensor, df_mask, "mittel")
real_images_tensor_big = get_mask_array(real_images_tensor, df_mask, "groß")

lama_images_tensor_small = get_mask_array(lama_images_tensor, df_mask, "klein")
lama_images_tensor_middle = get_mask_array(lama_images_tensor, df_mask, "mittel")
lama_images_tensor_big = get_mask_array(lama_images_tensor, df_mask, "groß")

opencv_images_tensor_small = get_mask_array(opencv_images_tensor, df_mask, "klein")
opencv_images_tensor_middle = get_mask_array(opencv_images_tensor, df_mask, "mittel")
opencv_images_tensor_big = get_mask_array(opencv_images_tensor, df_mask, "groß")

sd_images_tensor_small = get_mask_array(sd_images_tensor, df_mask, "klein")
sd_images_tensor_middle = get_mask_array(sd_images_tensor, df_mask, "mittel")
sd_images_tensor_big = get_mask_array(sd_images_tensor, df_mask, "groß")

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

with open("fid_scores.json", "w") as f:
    json.dump(fid_scores, f)

logger.info("FID scores saved to fid_scores.json")

