"""
image_loader.py - Charge des images PGM et prépare les données.
No external libraries: ONLY Python standard library.
"""

import os
from pathlib import Path

def load_pgm(filepath):
    """
    Charge une image PGM ASCII et retourne la matrice de pixels.
    Format: P2 (PGM ASCII)
    Retour: liste de listes [[pixels]]
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    idx = 0
    while lines[idx].startswith('#'):
        idx += 1
    
    magic = lines[idx].strip()
    assert magic == "P2", f"Expected P2 (PGM ASCII), got {magic}"
    
    idx += 1
    width, height = map(int, lines[idx].split())
    idx += 1
    max_val = int(lines[idx].strip())
    idx += 1
    
    # Parse pixel data
    pixels = []
    for line in lines[idx:]:
        if line.strip():
            pixels.extend(map(int, line.split()))
    
    # Reshape to 2D
    image = []
    for i in range(height):
        row = pixels[i*width:(i+1)*width]
        image.append(row)
    
    return image

def load_ppm(filepath):
    """
    Charge une image PPM ASCII et retourne la matrice de pixels RGB.
    Format: P3 (PPM ASCII RGB)
    Retour: liste de listes [[[R,G,B], ...], ...]
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    idx = 0
    while lines[idx].startswith('#'):
        idx += 1
    
    magic = lines[idx].strip()
    assert magic == "P3", f"Expected P3 (PPM ASCII), got {magic}"
    
    idx += 1
    width, height = map(int, lines[idx].split())
    idx += 1
    max_val = int(lines[idx].strip())
    idx += 1
    
    # Parse pixel data
    pixels = []
    for line in lines[idx:]:
        if line.strip():
            pixels.extend(map(int, line.split()))
    
    # Reshape to 3D [height][width][3]
    image = []
    for i in range(height):
        row = []
        for j in range(width):
            pixel_idx = (i * width + j) * 3
            pixel = [pixels[pixel_idx], pixels[pixel_idx+1], pixels[pixel_idx+2]]
            row.append(pixel)
        image.append(row)
    
    return image

def image_to_vector(image):
    """
    Convertit une image 2D en vecteur 1D et normalise [0, 255] -> [0, 1].
    """
    vector = []
    for row in image:
        for pixel in row:
            vector.append(pixel / 255.0)
    return vector

def ppm_to_vector(image):
    """
    Convertit une image PPM (RGB) en vecteur 1D et normalise [0, 255] -> [0, 1].
    Retourne: [R1, G1, B1, R2, G2, B2, ...] normalisé
    """
    vector = []
    for row in image:
        for pixel in row:  # pixel = [R, G, B]
            for channel in pixel:
                vector.append(channel / 255.0)
    return vector

def load_dataset(data_dir, split="train"):
    """
    Charge tout le dataset depuis le répertoire.
    
    Args:
        data_dir: path vers /data
        split: "train" ou "test"
    
    Retour:
        (images, labels, class_names)
        - images: liste de vecteurs normalisés
        - labels: liste d'indices (0, 1, 2)
        - class_names: ["happy", "neutral", "sad"]
    """
    images = []
    labels = []
    class_names = ["happy", "neutral", "sad"]
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    split_dir = os.path.join(data_dir, split)
    
    for class_name in class_names:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue
        
        # Charger toutes les images de cette classe
        pgm_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.pgm')])
        
        for pgm_file in pgm_files:
            filepath = os.path.join(class_dir, pgm_file)
            try:
                image = load_pgm(filepath)
                vector = image_to_vector(image)
                images.append(vector)
                labels.append(class_to_idx[class_name])
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return images, labels, class_names

def load_all_data(data_dir):
    """
    Charge train + test.
    
    Retour:
        (train_images, train_labels, test_images, test_labels, class_names)
    """
    train_images, train_labels, class_names = load_dataset(data_dir, "train")
    test_images, test_labels, _ = load_dataset(data_dir, "test")
    
    return train_images, train_labels, test_images, test_labels, class_names

def load_color_dataset(data_dir, split="train"):
    """
    Charge le dataset en couleur (PPM).
    
    Args:
        data_dir: path vers /data
        split: "train" ou "test"
    
    Retour:
        (images, labels, class_names)
    """
    images = []
    labels = []
    class_names = ["happy", "neutral", "sad", "angry", "surprised"]
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    split_dir = os.path.join(data_dir, split)
    
    for class_name in class_names:
        class_dir = os.path.join(split_dir, f'{class_name}_color')
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue
        
        # Charger toutes les images PPM de cette classe
        ppm_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.ppm')])
        
        for ppm_file in ppm_files:
            filepath = os.path.join(class_dir, ppm_file)
            try:
                image = load_ppm(filepath)
                vector = ppm_to_vector(image)
                images.append(vector)
                labels.append(class_to_idx[class_name])
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return images, labels, class_names

def load_all_color_data(data_dir):
    """
    Charge train + test en couleur.
    
    Retour:
        (train_images, train_labels, test_images, test_labels, class_names)
    """
    train_images, train_labels, class_names = load_color_dataset(data_dir, "train")
    test_images, test_labels, _ = load_color_dataset(data_dir, "test")
    
    return train_images, train_labels, test_images, test_labels, class_names
