#!/usr/bin/env python3
"""
train.py - Entraînement du réseau de neurones.
Étape A: Baseline simple

Usage:
    python3 src/train.py
"""

import os
import sys

# Ajoute le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from generate_data import generate_dataset
from image_loader import load_all_data
from model import NeuralNetwork
from metrics import print_metrics, export_metrics_json

def main():
    print("="*60)
    print("ÉTAPE A - Baseline Simple")
    print("="*60)
    
    # Chemins
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Étape 1: Générer les données
    print("\n[1/4] Generating synthetic dataset...")
    generate_dataset()
    
    # Étape 2: Charger les données
    print("\n[2/4] Loading dataset...")
    train_images, train_labels, test_images, test_labels, class_names = load_all_data(data_dir)
    
    print(f"  Train: {len(train_images)} images ({len(class_names)} classes)")
    print(f"  Test:  {len(test_images)} images")
    print(f"  Input size: {len(train_images[0])} (8×8 pixels)")
    print(f"  Classes: {class_names}")
    
    # Étape 3: Créer et entraîner le modèle
    print("\n[3/4] Training model...")
    
    # Hyperparamètres
    input_size = len(train_images[0])
    hidden_size = 8  # Petit modèle pour baseline
    output_size = len(class_names)
    learning_rate = 0.1
    num_epochs = 50
    
    # Créer le réseau
    model = NeuralNetwork(input_size, hidden_size, output_size, seed=42)
    
    print(f"  Architecture: {input_size} -> {hidden_size} (ReLU) -> {output_size} (Softmax)")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}\n")
    
    # Entraînement
    train_losses = []
    print("  Epoch | Train Loss")
    print("  " + "-"*25)
    
    for epoch in range(num_epochs):
        loss = model.train_one_epoch(train_images, train_labels, learning_rate)
        train_losses.append(loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  {epoch+1:5d} | {loss:.6f}")
    
    print(f"\n  Final train loss: {train_losses[-1]:.6f}")
    
    # Étape 4: Évaluation
    print("\n[4/4] Evaluation...")
    
    # Prédictions
    train_pred = [model.predict(x) for x in train_images]
    test_pred = [model.predict(x) for x in test_images]
    
    # Afficher les métriques
    print("\n--- TRAIN SET ---")
    train_matrix, train_acc, train_f1 = print_metrics(train_labels, train_pred, class_names)
    
    print("\n--- TEST SET ---")
    test_matrix, test_acc, test_f1 = print_metrics(test_labels, test_pred, class_names)
    
    # Exporter les résultats
    metrics_file = os.path.join(results_dir, 'baseline_metrics.json')
    export_metrics_json(test_labels, test_pred, class_names, metrics_file)
    
    # Sauvegarder les courbes de loss
    loss_file = os.path.join(results_dir, 'training_loss.csv')
    with open(loss_file, 'w') as f:
        f.write("epoch,train_loss\n")
        for epoch, loss in enumerate(train_losses):
            f.write(f"{epoch+1},{loss}\n")
    print(f"✓ Loss history exported to {loss_file}")
    
    print("\n" + "="*60)
    print("✓ ÉTAPE A Complete!")
    print(f"  Next: Run ÉTAPE B for proper validation split")
    print("="*60)

if __name__ == "__main__":
    main()
