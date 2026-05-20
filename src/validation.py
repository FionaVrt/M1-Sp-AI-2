"""
validation.py - Gestion du split train/val/test et validation croisée.
"""

import random

def train_val_test_split(images, labels, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split les données en train / validation / test.
    
    Args:
        images: liste des images (vecteurs)
        labels: liste des labels (indices)
        train_ratio: proportion train (défaut 0.7)
        val_ratio: proportion validation (défaut 0.15)
        seed: seed pour reproductibilité
    
    Retour:
        (train_images, train_labels, val_images, val_labels, test_images, test_labels)
    """
    random.seed(seed)
    
    # Créer une liste d'indices
    indices = list(range(len(images)))
    random.shuffle(indices)
    
    # Calculer les split points
    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Diviser
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # Créer les subsets
    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    
    val_images = [images[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    test_images = [images[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def kfold_split(images, labels, k=5, seed=42):
    """
    K-fold cross-validation split.
    
    Args:
        images: liste des images
        labels: liste des labels
        k: nombre de folds
        seed: seed
    
    Retour:
        Liste de k tuples (train_images, train_labels, val_images, val_labels)
    """
    random.seed(seed)
    
    indices = list(range(len(images)))
    random.shuffle(indices)
    
    # Diviser en k folds
    fold_size = len(images) // k
    folds = []
    
    for i in range(k):
        start = i * fold_size
        if i == k - 1:  # Dernier fold prend le reste
            end = len(images)
        else:
            end = (i + 1) * fold_size
        
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        
        train_images = [images[j] for j in train_idx]
        train_labels = [labels[j] for j in train_idx]
        val_images = [images[j] for j in val_idx]
        val_labels = [labels[j] for j in val_idx]
        
        folds.append((train_images, train_labels, val_images, val_labels))
    
    return folds

def stratified_split(images, labels, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split stratifié: préserve la distribution des classes dans chaque subset.
    
    Args:
        images: liste des images
        labels: liste des labels
        train_ratio: proportion train
        val_ratio: proportion validation
        seed: seed
    
    Retour:
        (train_images, train_labels, val_images, val_labels, test_images, test_labels)
    """
    random.seed(seed)
    
    # Grouper par classe
    class_indices = {}
    for i, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    # Shuffle et split chaque classe
    train_idx, val_idx, test_idx = [], [], []
    
    for class_id, indices in class_indices.items():
        random.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train+n_val])
        test_idx.extend(indices[n_train+n_val:])
    
    # Créer les subsets
    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    
    val_images = [images[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    test_images = [images[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels
