import os
import csv
import numpy as np
from PIL import Image

def mask_generator(masks_dir, mask_files):
    """
    Generator zum Laden der Maskenbilder.

    Args:
    - masks_dir (str): Verzeichnis der Masken.
    - mask_files (list): Liste der Masken-Dateinamen.

    Yields:
    - tuple: Dateiname und PIL Image der Maske.
    """
    for fname in mask_files:
        mask_path = os.path.join(masks_dir, fname)
        try:
            mask = Image.open(mask_path).convert('L')  # In Graustufen konvertieren
            yield fname, mask
        except Exception as e:
            print(f"Fehler beim Laden von {fname}: {e}")

def get_white_pixel_count(mask):
    """
    Berechnet die Anzahl weißer Pixel in einer Maske.

    Args:
    - mask (PIL Image): Maskenbild.

    Returns:
    - int: Anzahl der weißen Pixel.
    """
    mask = mask.convert('L')
    
    mask_array = np.array(mask)
    white_pixel_count = np.sum(mask_array == 255)  # Weiß hat den Wert 255
    return white_pixel_count

def save_mask_sizes_to_csv(masks_dir, output_csv):
    """
    Lädt Masken, berechnet ihre Größe und speichert die Ergebnisse in einer CSV-Datei.

    Args:
    - masks_dir (str): Verzeichnis der Maskenbilder.
    - output_csv (str): Pfad zur Ausgabe-CSV-Datei.
    """
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png') or f.endswith('.jpg')]
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Bildname", "Maskengröße (weiße Pixel)"])
        
        for fname, mask in mask_generator(masks_dir, mask_files):
            mask_size = get_white_pixel_count(mask)
            writer.writerow([fname, mask_size])
    
    print(f"CSV-Datei gespeichert unter: {output_csv}")




if __name__ == "__main__":
    masks_directory = "Dataset_new/masks"  # Pfad zu deinem Masken-Ordner
    output_csv_file = "mask_sizes.csv"  # Name der Ausgabedatei
    save_mask_sizes_to_csv(masks_directory, output_csv_file)



