#!/usr/bin/env python3
"""
train_etape_d.py - Extension à la couleur et 5 classes
Étape D: Comparaison noir/blanc vs couleur

Usage:
    python3 src/train_etape_d.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from generate_data_color import generate_color_dataset
from image_loader import load_all_data, load_all_color_data
from model import NeuralNetwork
from metrics import print_metrics, confusion_matrix, accuracy, precision_recall_f1
from validation import stratified_split

def train_and_evaluate(model, train_images, train_labels, val_images, val_labels,
                      test_images, test_labels, class_names, num_epochs=100):
    """Entraîne un modèle et évalue."""
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        train_loss = model.train_one_epoch(train_images, train_labels, learning_rate=0.1)
        val_loss = model.compute_loss(val_images, val_labels)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    # Évaluation
    test_pred = [model.predict(x) for x in test_images]
    test_acc = accuracy(test_labels, test_pred)
    
    # F1 macro
    matrix = confusion_matrix(test_labels, test_pred, len(class_names))
    f1_scores = []
    for class_id in range(len(class_names)):
        _, _, f1 = precision_recall_f1(matrix, class_id)
        f1_scores.append(f1)
    test_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    return {
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'test_acc': test_acc,
        'test_f1': test_f1,
        'matrix': matrix,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

def main():
    print("="*70)
    print("ÉTAPE D - Extension: Couleur et 5 classes")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # PARTIE 1: Noir et blanc, 3 classes (baseline)
    print("\n[1/5] BLACK & WHITE - 3 Classes (Baseline)")
    print("-"*70)
    
    print("Loading BW data...")
    train_img_bw, train_lbl_bw, test_img_bw, test_lbl_bw, classes_bw = load_all_data(data_dir)
    
    all_img_bw = train_img_bw + test_img_bw
    all_lbl_bw = train_lbl_bw + test_lbl_bw
    
    print(f"  Total: {len(all_img_bw)} images")
    print(f"  Classes: {classes_bw}")
    print(f"  Input size: {len(all_img_bw[0])}")
    
    # Split
    train_bw, lbl_train_bw, val_bw, lbl_val_bw, test_bw, lbl_test_bw = \
        stratified_split(all_img_bw, all_lbl_bw, seed=42)
    
    # Train
    print("Training model...")
    model_bw = NeuralNetwork(len(all_img_bw[0]), 8, len(classes_bw), seed=42)
    results_bw = train_and_evaluate(model_bw, train_bw, lbl_train_bw, val_bw, lbl_val_bw,
                                   test_bw, lbl_test_bw, classes_bw, num_epochs=100)
    
    print(f"  Test Accuracy: {results_bw['test_acc']:.4f}")
    print(f"  Test Macro F1: {results_bw['test_f1']:.4f}")
    print(f"  Train/Val loss: {results_bw['train_loss']:.6f} / {results_bw['val_loss']:.6f}")
    
    # PARTIE 2: Couleur, 5 classes
    print("\n[2/5] COLOR - 5 Classes")
    print("-"*70)
    
    print("Generating color dataset...")
    generate_color_dataset()
    
    print("Loading COLOR data...")
    train_img_color, train_lbl_color, test_img_color, test_lbl_color, classes_color = \
        load_all_color_data(data_dir)
    
    all_img_color = train_img_color + test_img_color
    all_lbl_color = train_lbl_color + test_lbl_color
    
    print(f"  Total: {len(all_img_color)} images")
    print(f"  Classes: {classes_color}")
    print(f"  Input size: {len(all_img_color[0])} (16×16×3 channels)")
    
    # Split
    train_color, lbl_train_color, val_color, lbl_val_color, test_color, lbl_test_color = \
        stratified_split(all_img_color, all_lbl_color, seed=42)
    
    # Train
    print("Training model...")
    input_size_color = len(all_img_color[0])
    model_color = NeuralNetwork(input_size_color, 16, len(classes_color), seed=42)
    results_color = train_and_evaluate(model_color, train_color, lbl_train_color,
                                      val_color, lbl_val_color, test_color, lbl_test_color,
                                      classes_color, num_epochs=100)
    
    print(f"  Test Accuracy: {results_color['test_acc']:.4f}")
    print(f"  Test Macro F1: {results_color['test_f1']:.4f}")
    print(f"  Train/Val loss: {results_color['train_loss']:.6f} / {results_color['val_loss']:.6f}")
    
    # COMPARAISON
    print("\n[3/5] COMPARISON")
    print("="*70)
    
    print(f"\n{'Métrique':<25} {'B&W (3 classes)':<20} {'Color (5 classes)':<20}")
    print("-"*70)
    print(f"{'Input size':<25} {len(train_bw[0]):<20} {len(train_color[0]):<20}")
    print(f"{'Number of classes':<25} {len(classes_bw):<20} {len(classes_color):<20}")
    print(f"{'Test accuracy':<25} {results_bw['test_acc']:<20.4f} {results_color['test_acc']:<20.4f}")
    print(f"{'Test Macro F1':<25} {results_bw['test_f1']:<20.4f} {results_color['test_f1']:<20.4f}")
    print(f"{'Final train loss':<25} {results_bw['train_loss']:<20.6f} {results_color['train_loss']:<20.6f}")
    print(f"{'Final val loss':<25} {results_bw['val_loss']:<20.6f} {results_color['val_loss']:<20.6f}")
    
    # ANALYSE
    print("\n[4/5] ANALYSIS")
    print("="*70)
    
    print("\nColor vs B&W Analysis:")
    print("-" * 70)
    
    if results_color['test_acc'] > results_bw['test_acc']:
        acc_diff = (results_color['test_acc'] - results_bw['test_acc']) * 100
        print(f"✓ Color model is BETTER: +{acc_diff:.2f}% accuracy")
        print("  → La couleur porte une information utile pour distinguer les classes")
    elif results_color['test_acc'] < results_bw['test_acc']:
        acc_diff = (results_bw['test_acc'] - results_color['test_acc']) * 100
        print(f"✗ Color model is WORSE: -{acc_diff:.2f}% accuracy")
        print("  → La couleur crée du bruit ou de l'overfitting")
    else:
        print("≈ Color and B&W have SAME accuracy")
        print("  → La couleur ne change rien pour cette tâche")
    
    print(f"\nClasses added (Angry, Surprised):")
    for class_id, class_name in enumerate(classes_color):
        if class_name not in classes_bw:
            print(f"  - {class_name}: Newly added")
    
    # Export
    print("\n[5/5] Exporting results...")
    
    export_dict = {
        'bw_model': {
            'num_classes': len(classes_bw),
            'input_size': len(train_bw[0]),
            'test_accuracy': results_bw['test_acc'],
            'test_f1': results_bw['test_f1'],
            'train_loss_final': results_bw['train_loss'],
            'val_loss_final': results_bw['val_loss'],
            'classes': classes_bw
        },
        'color_model': {
            'num_classes': len(classes_color),
            'input_size': len(train_color[0]),
            'test_accuracy': results_color['test_acc'],
            'test_f1': results_color['test_f1'],
            'train_loss_final': results_color['train_loss'],
            'val_loss_final': results_color['val_loss'],
            'classes': classes_color
        },
        'comparison': {
            'accuracy_difference': results_color['test_acc'] - results_bw['test_acc'],
            'f1_difference': results_color['test_f1'] - results_bw['test_f1'],
            'color_better': results_color['test_acc'] > results_bw['test_acc']
        }
    }
    
    metrics_file = os.path.join(results_dir, 'etape_d_color_analysis.json')
    with open(metrics_file, 'w') as f:
        json.dump(export_dict, f, indent=2)
    
    print(f"✓ Analysis exported to {metrics_file}")
    
    print("\n" + "="*70)
    print("✓ ÉTAPE D Complete!")
    print("  - B&W model trained (3 classes)")
    print("  - Color model trained (5 classes)")
    print("  - Comparison saved")
    print("  Next: Write the final report")
    print("="*70)

if __name__ == "__main__":
    main()
