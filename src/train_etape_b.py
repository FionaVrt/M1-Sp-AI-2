#!/usr/bin/env python3
"""
train_etape_b.py - Entraînement avec validation proper
Étape B: Train / Validation / Test split

Usage:
    python3 src/train_etape_b.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_loader import load_all_data
from model import NeuralNetwork
from metrics import print_metrics, confusion_matrix, accuracy, precision_recall_f1, macro_f1
from validation import stratified_split

def main():
    print("="*60)
    print("ÉTAPE B - Évaluation Propre (Train/Val/Test)")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Charger les données (combine train + test de l'ÉTAPE A)
    print("\n[1/5] Loading all data...")
    train_images, train_labels, test_images, test_labels, class_names = load_all_data(data_dir)
    
    # Combiner
    all_images = train_images + test_images
    all_labels = train_labels + test_labels
    
    print(f"  Total images: {len(all_images)}")
    print(f"  Classes: {class_names}")
    
    # Stratified split: 70/15/15
    print("\n[2/5] Stratified split (70% train / 15% val / 15% test)...")
    train_images, train_labels, val_images, val_labels, test_images, test_labels = \
        stratified_split(all_images, all_labels, train_ratio=0.7, val_ratio=0.15, seed=42)
    
    print(f"  Train: {len(train_images)} images")
    print(f"  Val:   {len(val_images)} images")
    print(f"  Test:  {len(test_images)} images")
    
    # Création du modèle
    print("\n[3/5] Training model with validation monitoring...")
    
    input_size = len(train_images[0])
    hidden_size = 8
    output_size = len(class_names)
    learning_rate = 0.1
    num_epochs = 150
    
    model = NeuralNetwork(input_size, hidden_size, output_size, seed=42)
    
    print(f"  Architecture: {input_size} -> {hidden_size} (ReLU) -> {output_size} (Softmax)")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}\n")
    
    # Entraînement avec monitoring validation
    train_losses = []
    val_losses = []
    
    print("  Epoch | Train Loss | Val Loss")
    print("  " + "-"*40)
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss = model.train_one_epoch(train_images, train_labels, learning_rate)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = model.compute_loss(val_images, val_labels, regularization=False)
        val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  {epoch+1:5d} | {train_loss:.6f}  | {val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (val loss not improving)")
            break
    
    print(f"\n  Final train loss: {train_losses[-1]:.6f}")
    print(f"  Final val loss:   {val_losses[-1]:.6f}")
    
    # Évaluation
    print("\n[4/5] Evaluation on test set...")
    
    test_pred = [model.predict(x) for x in test_images]
    
    print("\n--- TEST SET RESULTS ---")
    test_matrix, test_acc, test_f1 = print_metrics(test_labels, test_pred, class_names)
    
    # Export metrics
    print("\n[5/5] Exporting results...")
    
    # Metrics JSON
    import json
    metrics_file = os.path.join(results_dir, 'etape_b_metrics.json')
    metrics_dict = {
        'split': {
            'train_size': len(train_images),
            'val_size': len(val_images),
            'test_size': len(test_images)
        },
        'test_accuracy': test_acc,
        'test_macro_f1': test_f1,
        'confusion_matrix': test_matrix,
        'final_losses': {
            'train': train_losses[-1],
            'val': val_losses[-1]
        }
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"✓ Metrics exported to {metrics_file}")
    
    # Loss history
    loss_file = os.path.join(results_dir, 'etape_b_loss.csv')
    with open(loss_file, 'w') as f:
        f.write("epoch,train_loss,val_loss\n")
        for epoch, (tloss, vloss) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{epoch+1},{tloss:.6f},{vloss:.6f}\n")
    print(f"✓ Loss history exported to {loss_file}")
    
    print("\n" + "="*60)
    print("✓ ÉTAPE B Complete!")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Macro F1: {test_f1:.4f}")
    print(f"  Next: Run ÉTAPE C for overfitting analysis")
    print("="*60)

if __name__ == "__main__":
    main()
