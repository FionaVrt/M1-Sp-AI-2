#!/usr/bin/env python3


import os
import random

# Configuration
SIZE = 8
MAX_VAL = 255
NUM_TRAIN_PER_CLASS = 25
NUM_TEST_PER_CLASS = 8

def create_blank_image(size=SIZE):
    """Crée une image blanche (0 = noir, 255 = blanc)."""
    return [[255] * size for _ in range(size)]

def draw_circle(image, cx, cy, radius, color=0):
    """Dessine un cercle rempli."""
    size = len(image)
    for y in range(size):
        for x in range(size):
            dx = x - cx
            dy = y - cy
            if dx*dx + dy*dy <= radius*radius:
                image[y][x] = color

def draw_line(image, x1, y1, x2, y2, color=0, thickness=1):
    """Dessine une ligne simple (algorithme de Bresenham basique)."""
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
                        image[y+dy][x+dx] = color

def create_happy(variation=0):
    """Crée un smiley Happy (sourire)."""
    img = create_blank_image()
    
    # Yeux
    eye_y = 2
    left_eye_x = 2
    right_eye_x = 6
    
    # Variation de position
    offset = variation % 3
    left_eye_x += offset - 1
    right_eye_x -= offset - 1
    
    if 0 <= left_eye_x < SIZE and 0 <= right_eye_x < SIZE:
        draw_circle(img, left_eye_x, eye_y, 1, color=0)
        draw_circle(img, right_eye_x, eye_y, 1, color=0)
    
    # Bouche (sourire courbe vers le haut)
    mouth_y = 6
    mouth_width = variation % 2 + 1  # Épaisseur légère
    draw_line(img, 2, 6, 6, 5, color=0, thickness=mouth_width)
    draw_line(img, 6, 5, 6, 6, color=0, thickness=mouth_width)
    
    return img

def create_neutral(variation=0):
    """Crée un smiley Neutral (bouche droite)."""
    img = create_blank_image()
    
    # Yeux
    eye_y = 2
    left_eye_x = 2 + (variation % 2)
    right_eye_x = 6 - (variation % 2)
    
    if 0 <= left_eye_x < SIZE and 0 <= right_eye_x < SIZE:
        draw_circle(img, left_eye_x, eye_y, 1, color=0)
        draw_circle(img, right_eye_x, eye_y, 1, color=0)
    
    # Bouche (ligne droite)
    mouth_y = 6
    mouth_thickness = 1
    draw_line(img, 2, mouth_y, 6, mouth_y, color=0, thickness=mouth_thickness)
    
    return img

def create_sad(variation=0):
    """Crée un smiley Sad (bouche vers le bas)."""
    img = create_blank_image()
    
    # Yeux
    eye_y = 2
    left_eye_x = 2 + (variation % 2)
    right_eye_x = 6 - (variation % 2)
    
    if 0 <= left_eye_x < SIZE and 0 <= right_eye_x < SIZE:
        draw_circle(img, left_eye_x, eye_y, 1, color=0)
        draw_circle(img, right_eye_x, eye_y, 1, color=0)
    
    # Bouche (sourire inversé vers le bas)
    mouth_y = 6
    mouth_width = variation % 2 + 1
    draw_line(img, 2, 5, 6, 6, color=0, thickness=mouth_width)
    draw_line(img, 2, 5, 2, 6, color=0, thickness=mouth_width)
    
    return img

def save_pgm(image, filepath):
    """Sauvegarde une image au format PGM ASCII."""
    size = len(image)
    with open(filepath, 'w') as f:
        f.write("P2\n")  # PGM ASCII
        f.write(f"{size} {size}\n")
        f.write(f"{MAX_VAL}\n")
        for row in image:
            f.write(" ".join(map(str, row)) + "\n")

def generate_dataset():
    """Génère le dataset complet."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Train set
    print("Générating TRAIN set...")
    for i in range(NUM_TRAIN_PER_CLASS):
        # Happy
        img = create_happy(variation=i)
        path = os.path.join(base_path, 'train', 'happy', f'happy_{i:03d}.pgm')
        save_pgm(img, path)
        
        # Neutral
        img = create_neutral(variation=i)
        path = os.path.join(base_path, 'train', 'neutral', f'neutral_{i:03d}.pgm')
        save_pgm(img, path)
        
        # Sad
        img = create_sad(variation=i)
        path = os.path.join(base_path, 'train', 'sad', f'sad_{i:03d}.pgm')
        save_pgm(img, path)
    
    # Test set
    print("Generating TEST set...")
    for i in range(NUM_TEST_PER_CLASS):
        # Happy
        img = create_happy(variation=NUM_TRAIN_PER_CLASS + i)
        path = os.path.join(base_path, 'test', 'happy', f'happy_test_{i:03d}.pgm')
        save_pgm(img, path)
        
        # Neutral
        img = create_neutral(variation=NUM_TRAIN_PER_CLASS + i)
        path = os.path.join(base_path, 'test', 'neutral', f'neutral_test_{i:03d}.pgm')
        save_pgm(img, path)
        
        # Sad
        img = create_sad(variation=NUM_TRAIN_PER_CLASS + i)
        path = os.path.join(base_path, 'test', 'sad', f'sad_test_{i:03d}.pgm')
        save_pgm(img, path)
    
    print(f"✓ Dataset créé:")
    print(f"  - Train: {NUM_TRAIN_PER_CLASS} images par classe (3 classes)")
    print(f"  - Test:  {NUM_TEST_PER_CLASS} images par classe (3 classes)")
    print(f"  - Format: PGM ASCII 8×8")
    print(f"  - Location: {base_path}")

if __name__ == "__main__":
    generate_dataset()
