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

if __name__ == "__main__":
    # Command-Line Argumente
    parser = argparse.ArgumentParser(description="Calculate FID scores for image sets")
    parser.add_argument(
        '--manual_file', 
        type=str, 
        default=None, 
        help="Pfad zu einer Textdatei mit Bildnamen (eine pro Zeile), die anstelle des Schnittmengen-Sets verwendet werden."
    )
    parser.add_argument(
        '--unique_count',
        type=int,
        default=100,
        help="Anzahl der einzigartigen Bilder."
    )
    parser.add_argument(
        '--duplicate_count',
        type=int,
        default=900,
        help="Anzahl der Duplikate, die aus den einzigartigen Bildern hinzugefügt werden sollen."
    )
    args = parser.parse_args()

    # Verzeichnisse
    images_dir = './Dataset_new/images'
    masks_dir = './Dataset_new/masks'
    results_dir_lama = 'results/lama'
    results_dir_opencv = 'results/opencv'
    results_dir_sd = 'results/sd'

    # Liste der Dateien in den Verzeichnissen
    real_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') and f in os.listdir(masks_dir)]
    lama_files = [f for f in os.listdir(results_dir_lama) if f.endswith('.jpg')]
    opencv_files = [f for f in os.listdir(results_dir_opencv) if f.endswith('.jpg')]
    sd_files = [f for f in os.listdir(results_dir_sd) if f.endswith('.jpg')]

    # Schnittmenge der Dateien, die in allen Verzeichnissen vorhanden sind
    image_files = list(set(real_files) & set(lama_files) & set(opencv_files) & set(sd_files))
    image_files.sort()  # für konsistente Reihenfolge
    logger.info(f"Found {len(image_files)} images in all directories")

    # Falls eine manuelle Datei übergeben wurde, wird diese Liste verwendet
    if args.manual_file is not None:
        if os.path.exists(args.manual_file):
            with open(args.manual_file, 'r') as f:
                manual_list = [line.strip() for line in f if line.strip()]
            image_files = list(set(image_files) & set(manual_list))
            image_files.sort()
            logger.info(f"Manual file provided. {len(image_files)} images remain after filtering.")
        else:
            logger.error(f"Manual file {args.manual_file} does not exist. Exiting.")
            exit(1)

    # Sicherstellen, dass genügend einzigartige Bilder vorhanden sind
    if len(image_files) < args.unique_count:
        logger.error(f"Not enough images available. Required {args.unique_count} unique images, but found {len(image_files)}.")
        exit(1)

    # Auswahl der 600 einzigartigen Bilder
    unique_images = image_files[:args.unique_count]

    # Erzeugen der Duplikate: Hier werden z.B. die ersten 400 Bilder aus den einzigartigen kopiert.
    # Falls duplicate_count größer als die unique_images ist, wird per Zufall (mit Zurücklegen) ausgewählt.
    if args.duplicate_count > len(unique_images):
        logger.warning("Duplicate count is greater than the number of unique images. Sampling with replacement.")
        duplicate_images = list(np.random.choice(unique_images, size=args.duplicate_count, replace=True))
    else:
        duplicate_images = unique_images[:args.duplicate_count]

    # Endgültige Liste: 600 unique + 400 Duplicate = 1000 Bilder
    final_image_files = unique_images + duplicate_images
    logger.info(f"Using {args.unique_count} unique images and {args.duplicate_count} duplicates, total {len(final_image_files)} images.")

    # Feste Zielgröße für die Vorverarbeitung
    target_size = (256, 256)

    # Laden der Bilder mittels image_generator
    logger.info("Loading images...")
    real_images = list(image_generator(images_dir, final_image_files))
    lama_images = list(image_generator(results_dir_lama, final_image_files))
    opencv_images = list(image_generator(results_dir_opencv, final_image_files))
    sd_images = list(image_generator(results_dir_sd, final_image_files))

    logger.info(f"Loaded {len(real_images)} Real images")
    logger.info(f"Loaded {len(lama_images)} Lama images")
    logger.info(f"Loaded {len(opencv_images)} OpenCV images")
    logger.info(f"Loaded {len(sd_images)} SD images")

    # Vorverarbeitung (z.B. Größenanpassung, Normalisierung)
    real_images_tensor = preprocess_images(real_images, target_size)
    lama_images_tensor = preprocess_images(lama_images, target_size)
    opencv_images_tensor = preprocess_images(opencv_images, target_size)
    sd_images_tensor = preprocess_images(sd_images, target_size)

    # Berechnung der FID-Scores zwischen den realen Bildern und den generierten Bildern
    fid_score_lama = calculate_fid(real_images_tensor, lama_images_tensor)
    fid_score_opencv = calculate_fid(real_images_tensor, opencv_images_tensor)
    fid_score_sd = calculate_fid(real_images_tensor, sd_images_tensor)

    logger.info(f"FID Lama: {fid_score_lama}")
    logger.info(f"FID OpenCV: {fid_score_opencv}")
    logger.info(f"FID SD: {fid_score_sd}")

    # Speichern der FID-Scores in einer JSON-Datei
    fid_scores = {
        "lama": fid_score_lama,
        "opencv": fid_score_opencv,
        "sd": fid_score_sd
    }
    output_file = f"get_fid_{len(final_image_files)}_images.json"
    with open(output_file, "w") as f:
        json.dump(fid_scores, f)

    logger.info(f"FID scores saved to {output_file}")

