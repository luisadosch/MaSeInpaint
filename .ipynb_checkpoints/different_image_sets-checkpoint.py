import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from statistical_tests import image_generator, preprocess_images, calculate_fid, bootstrap_fid, check_normality, wilcoxon_signed_rank_test, two_sided_t_test

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional helper: get a subset of images (this could be expanded if needed)
def get_image_subset(image_list, subset_size):
    """Return a subset of images based on the desired subset size."""
    # Here you could also implement custom logic (e.g., random sampling)
    return image_list[:subset_size]

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Calculate FID scores for image sets")
    parser.add_argument(
        '--subset_size', 
        type=int, 
        default=100, 
        help="Number of images to use from the intersection of available images. Use 0 or a negative number to use all images."
    )
    parser.add_argument(
        '--manual_file', 
        type=str, 
        default=None, 
        help="Optional path to a text file listing image filenames (one per line) to use instead of the intersection."
    )
    args = parser.parse_args()

    # Directories
    images_dir = './Dataset_new/images'
    masks_dir = './Dataset_new/masks'
    results_dir_lama = 'results/lama'
    results_dir_opencv = 'results/opencv'
    results_dir_sd = 'results/sd'

    # List files in each directory
    real_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') and f in os.listdir(masks_dir)]
    lama_files = [f for f in os.listdir(results_dir_lama) if f.endswith('.jpg')]
    opencv_files = [f for f in os.listdir(results_dir_opencv) if f.endswith('.jpg')]
    sd_files = [f for f in os.listdir(results_dir_sd) if f.endswith('.jpg')]

    # Find the intersection of files available in all directories
    image_files = list(set(real_files) & set(lama_files) & set(opencv_files) & set(sd_files))
    image_files.sort()  # sort to ensure consistent ordering
    logger.info(f"Found {len(image_files)} images in all directories")

    # If a manual file is provided, override the image_files list
    if args.manual_file is not None:
        if os.path.exists(args.manual_file):
            with open(args.manual_file, 'r') as f:
                manual_list = [line.strip() for line in f if line.strip()]
            # Use only the files that are in the intersection and in your manual list
            image_files = list(set(image_files) & set(manual_list))
            image_files.sort()
            logger.info(f"Manual file provided. {len(image_files)} images remain after filtering.")
        else:
            logger.error(f"Manual file {args.manual_file} does not exist. Exiting.")
            exit(1)

    # Limit the number of images if subset_size is set to a positive value
    if args.subset_size > 0 and args.subset_size < len(image_files):
        image_files = get_image_subset(image_files, args.subset_size)
        logger.info(f"Using a subset of {len(image_files)} images.")

    # Define a fixed image size for preprocessing
    target_size = (256, 256)

    # Load the images using your image_generator
    logger.info("Loading images...")
    real_images = list(image_generator(images_dir, image_files))
    lama_images = list(image_generator(results_dir_lama, image_files))
    opencv_images = list(image_generator(results_dir_opencv, image_files))
    sd_images = list(image_generator(results_dir_sd, image_files))

    logger.info(f"Loaded {len(real_images)} Real images")
    logger.info(f"Loaded {len(lama_images)} Lama images")
    logger.info(f"Loaded {len(opencv_images)} OpenCV images")
    logger.info(f"Loaded {len(sd_images)} SD images")

    # Preprocess the images (e.g., resizing, normalization, tensor conversion)
    real_images_tensor = preprocess_images(real_images, target_size)
    lama_images_tensor = preprocess_images(lama_images, target_size)
    opencv_images_tensor = preprocess_images(opencv_images, target_size)
    sd_images_tensor = preprocess_images(sd_images, target_size)

    # Calculate FID scores between the real images and each set of generated images
    fid_score_lama = calculate_fid(real_images_tensor, lama_images_tensor)
    fid_score_opencv = calculate_fid(real_images_tensor, opencv_images_tensor)
    fid_score_sd = calculate_fid(real_images_tensor, sd_images_tensor)

    logger.info(f"FID Lama: {fid_score_lama}")
    logger.info(f"FID OpenCV: {fid_score_opencv}")
    logger.info(f"FID SD: {fid_score_sd}")

    # Save the FID scores to a JSON file (you might update the filename based on subset size/scenario)
    fid_scores = {
        "lama": fid_score_lama,
        "opencv": fid_score_opencv,
        "sd": fid_score_sd
    }
    output_file = f"get_fid_{len(image_files)}_images.json"
    with open(output_file, "w") as f:
        json.dump(fid_scores, f)

    logger.info(f"FID scores saved to {output_file}")


   
    