#!/usr/bin/env python3
"""
generate_data_color.py - Génère des smileys synthétiques en couleur (PPM)

Les 5 émotions:
  - Happy (0): rouge/sourire
  - Neutral (1): gris/bouche droite
  - Sad (2): bleu/bouche vers le bas
  - Angry (3): rouge foncé/sourcils froncés
  - Surprised (4): jaune/bouche ouverte
"""

import os
import random

SIZE = 16
MAX_VAL = 255
NUM_TRAIN_PER_CLASS = 20
NUM_TEST_PER_CLASS = 6

def create_blank_image(size=SIZE):
    """Crée une image RGB blanche."""
    return [[[255, 255, 255] for _ in range(size)] for _ in range(size)]

def draw_circle_rgb(image, cx, cy, radius, color=(0, 0, 0)):
    """Dessine un cercle rempli en couleur RGB."""
    size = len(image)
    for y in range(size):
        for x in range(size):
            dx = x - cx
            dy = y - cy
            if dx*dx + dy*dy <= radius*radius:
                image[y][x] = list(color)

def draw_line_rgb(image, x1, y1, x2, y2, color=(0, 0, 0), thickness=1):
    """Dessine une ligne en couleur RGB."""
    size = len(image)
    steps = max(abs(x2-x1), abs(y2-y1))
    if steps == 0:
        steps = 1
    for i in range(steps+1):
        t = i / steps
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        if 0 <= x < size and 0 <= y < size:
            for dx in range(-thickness, thickness+1):
                for dy in range(-thickness, thickness+1):
                    if 0 <= x+dx < size and 0 <= y+dy < size:
                        image[y+dy][x+dx] = list(color)

def fill_rect_rgb(image, x1, y1, x2, y2, color=(0, 0, 0)):
    """Remplit un rectangle."""
    for y in range(max(0, y1), min(len(image), y2+1)):
        for x in range(max(0, x1), min(len(image[0]), x2+1)):
            image[y][x] = list(color)

def create_happy(variation=0):
    """Happy: fond rose/rouge, sourire"""
    img = create_blank_image()
    
    # Fond rose/rouge clair
    color_bg = (255, 150 + variation % 50, 150)
    for y in range(SIZE):
        for x in range(SIZE):
            img[y][x] = list(color_bg)
    
    # Yeux noirs
    draw_circle_rgb(img, 4 + variation % 2, 5, 1, color=(0, 0, 0))
    draw_circle_rgb(img, 12 - variation % 2, 5, 1, color=(0, 0, 0))
    
    # Sourire courbe (rouge foncé)
    draw_line_rgb(img, 4, 11, 12, 10, color=(200, 0, 0), thickness=1)
    
    return img

def create_neutral(variation=0):
    """Neutral: fond gris, bouche droite"""
    img = create_blank_image()
    
    # Fond gris
    color_bg = (200 + variation % 20, 200 + variation % 20, 200 + variation % 20)
    for y in range(SIZE):
        for x in range(SIZE):
            img[y][x] = list(color_bg)
    
    # Yeux noirs
    draw_circle_rgb(img, 4 + variation % 2, 5, 1, color=(0, 0, 0))
    draw_circle_rgb(img, 12 - variation % 2, 5, 1, color=(0, 0, 0))
    
    # Bouche droite (noire)
    draw_line_rgb(img, 4, 11, 12, 11, color=(0, 0, 0), thickness=1)
    
    return img

def create_sad(variation=0):
    """Sad: fond bleu, bouche vers le bas"""
    img = create_blank_image()
    
    # Fond bleu clair
    color_bg = (150 + variation % 30, 180 + variation % 30, 255)
    for y in range(SIZE):
        for x in range(SIZE):
            img[y][x] = list(color_bg)
    
    # Yeux noirs
    draw_circle_rgb(img, 4 + variation % 2, 5, 1, color=(0, 0, 0))
    draw_circle_rgb(img, 12 - variation % 2, 5, 1, color=(0, 0, 0))
    
    # Bouche vers le bas (bleu foncé)
    draw_line_rgb(img, 4, 10, 12, 11, color=(0, 0, 150), thickness=1)
    
    return img

def create_angry(variation=0):
    """Angry: fond rouge foncé, sourcils froncés"""
    img = create_blank_image()
    
    # Fond rouge foncé
    color_bg = (255, 80 + variation % 40, 80 + variation % 40)
    for y in range(SIZE):
        for x in range(SIZE):
            img[y][x] = list(color_bg)
    
    # Sourcils froncés (noirs)
    draw_line_rgb(img, 2, 3, 6, 2, color=(0, 0, 0), thickness=1)
    draw_line_rgb(img, 10, 2, 14, 3, color=(0, 0, 0), thickness=1)
    
    # Yeux rouges agressifs
    draw_circle_rgb(img, 4, 6, 1, color=(255, 0, 0))
    draw_circle_rgb(img, 12, 6, 1, color=(255, 0, 0))
    
    # Bouche agressive (ligne vers le bas)
    draw_line_rgb(img, 4, 10, 12, 12, color=(0, 0, 0), thickness=2)
    
    return img

def create_surprised(variation=0):
    """Surprised: fond jaune, grande bouche ouverte"""
    img = create_blank_image()
    
    # Fond jaune
    color_bg = (255, 255, 100 + variation % 50)
    for y in range(SIZE):
        for x in range(SIZE):
            img[y][x] = list(color_bg)
    
    # Yeux ouverts (grand)
    draw_circle_rgb(img, 4, 5, 2, color=(0, 0, 0))
    draw_circle_rgb(img, 12, 5, 2, color=(0, 0, 0))
    
    # Bouche ouverte (grand O)
    draw_circle_rgb(img, 8, 11, 2, color=(0, 0, 0))
    
    return img

def save_ppm(image, filepath):
    """Sauvegarde une image RGB au format PPM ASCII."""
    size = len(image)
    with open(filepath, 'w') as f:
        f.write("P3\n")  # PPM ASCII RGB
        f.write(f"{size} {size}\n")
        f.write(f"{MAX_VAL}\n")
        for row in image:
            for pixel in row:
                f.write(f"{pixel[0]} {pixel[1]} {pixel[2]} ")
            f.write("\n")

def generate_color_dataset():
    """Génère le dataset synthétique en couleur avec 5 classes."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Créer les dossiers pour 5 classes
    classes = ['happy', 'neutral', 'sad', 'angry', 'surprised']
    
    for split in ['train', 'test']:
        for class_name in classes:
            class_dir = os.path.join(base_path, split, f'{class_name}_color')
            os.makedirs(class_dir, exist_ok=True)
    
    # Generateurs pour chaque classe
    generators = {
        'happy': create_happy,
        'neutral': create_neutral,
        'sad': create_sad,
        'angry': create_angry,
        'surprised': create_surprised,
    }
    
    # Train set
    print("Generating COLOR TRAIN set (5 classes)...")
    for class_name, generator in generators.items():
        for i in range(NUM_TRAIN_PER_CLASS):
            img = generator(variation=i)
            path = os.path.join(base_path, 'train', f'{class_name}_color', f'{class_name}_{i:03d}.ppm')
            save_ppm(img, path)
    
    # Test set
    print("Generating COLOR TEST set (5 classes)...")
    for class_name, generator in generators.items():
        for i in range(NUM_TEST_PER_CLASS):
            img = generator(variation=NUM_TRAIN_PER_CLASS + i)
            path = os.path.join(base_path, 'test', f'{class_name}_color', f'{class_name}_test_{i:03d}.ppm')
            save_ppm(img, path)
    
    print(f"✓ Color dataset créé:")
    print(f"  - Train: {NUM_TRAIN_PER_CLASS} images par classe × 5 classes")
    print(f"  - Test:  {NUM_TEST_PER_CLASS} images par classe × 5 classes")
    print(f"  - Format: PPM ASCII 16×16×RGB")
    print(f"  - Location: {base_path}")

if __name__ == "__main__":
    generate_color_dataset()
