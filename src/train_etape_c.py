#!/usr/bin/env python3
"""
train_etape_c.py - Analyse de l'overfitting
Étape C: Compare petit vs grand modèle

Usage:
    python3 src/train_etape_c.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_loader import load_all_data
from model import NeuralNetwork
from metrics import accuracy, macro_f1
from validation import stratified_split

def train_model_and_monitor(model, train_images, train_labels, val_images, val_labels,
                            learning_rate=0.1, num_epochs=100):
    """
    Entraîne le modèle et retourne les historiques train/val.
    """
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        train_loss = model.train_one_epoch(train_images, train_labels, learning_rate)
        val_loss = model.compute_loss(val_images, val_labels, regularization=False)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    return train_losses, val_losses

def evaluate_model(model, test_images, test_labels):
    """Évalue le modèle et retourne les métriques."""
    test_pred = [model.predict(x) for x in test_images]
    acc = accuracy(test_labels, test_pred)
    f1 = macro_f1([[0]*len(set(test_labels)) for _ in range(len(set(test_labels)))], [])  # Placeholder
    
    # Calcul réel du F1
    from metrics import confusion_matrix, precision_recall_f1
    matrix = confusion_matrix(test_labels, test_pred, len(set(test_labels)))
    f1_scores = []
    for class_id in range(len(set(test_labels))):
        _, _, f1_score = precision_recall_f1(matrix, class_id)
        f1_scores.append(f1_score)
    f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    return acc, f1

def main():
    print("="*70)
    print("ÉTAPE C - Analyse de l'Overfitting (Petit vs Grand Modèle)")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Charger données
    print("\n[1/5] Loading data...")
    train_images, train_labels, test_images, test_labels, class_names = load_all_data(data_dir)
    all_images = train_images + test_images
    all_labels = train_labels + test_labels
    
    print(f"  Total: {len(all_images)} images")
    
    # Split
    print("[2/5] Splitting data...")
    train_images, train_labels, val_images, val_labels, test_images, test_labels = \
        stratified_split(all_images, all_labels, train_ratio=0.7, val_ratio=0.15, seed=42)
    
    print(f"  Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Configurations à tester
    configs = [
        {'name': 'Small', 'hidden': 4, 'learning_rate': 0.1},
        {'name': 'Medium', 'hidden': 8, 'learning_rate': 0.1},
        {'name': 'Large', 'hidden': 32, 'learning_rate': 0.1},
        {'name': 'XL', 'hidden': 64, 'learning_rate': 0.05},
    ]
    
    print(f"\n[3/5] Training {len(configs)} models...\n")
    
    results = {}
    
    for config in configs:
        name = config['name']
        hidden_size = config['hidden']
        learning_rate = config['learning_rate']
        
        print(f"  Training {name} (hidden={hidden_size})...")
        
        input_size = len(train_images[0])
        output_size = len(class_names)
        
        model = NeuralNetwork(input_size, hidden_size, output_size, seed=42)
        
        # Entraîner
        train_losses, val_losses = train_model_and_monitor(
            model, train_images, train_labels, val_images, val_labels,
            learning_rate=learning_rate, num_epochs=100
        )
        
        # Évaluer
        test_acc, test_f1 = evaluate_model(model, test_images, test_labels)
        
        # Analyser overfitting (final loss gap)
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        overfit_gap = final_val_loss - final_train_loss if final_train_loss > 0 else 0
        is_overfitting = "YES" if overfit_gap > 0.01 else "NO"
        
        results[name] = {
            'hidden_size': hidden_size,
            'train_loss_final': final_train_loss,
            'val_loss_final': final_val_loss,
            'overfit_gap': overfit_gap,
            'is_overfitting': is_overfitting,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"    ✓ Train loss: {final_train_loss:.6f}, Val loss: {final_val_loss:.6f}")
        print(f"    ✓ Test accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"    ✓ Overfitting: {is_overfitting} (gap: {overfit_gap:.6f})\n")
    
    # Afficher comparaison
    print("[4/5] Comparison summary\n")
    print(f"{'Model':<10} {'Hidden':<8} {'Train Loss':<12} {'Val Loss':<12} {'Overfit?':<10} {'Test Acc':<10}")
    print("-"*70)
    
    for name in ['Small', 'Medium', 'Large', 'XL']:
        if name in results:
            r = results[name]
            print(f"{name:<10} {r['hidden_size']:<8} {r['train_loss_final']:<12.6f} "
                  f"{r['val_loss_final']:<12.6f} {r['is_overfitting']:<10} {r['test_accuracy']:<10.4f}")
    
    # Analyser
    print("\nAnalyse de l'overfitting:")
    print("-" * 70)
    
    for name, r in results.items():
        print(f"\n{name} (hidden={r['hidden_size']}):")
        print(f"  - Train loss final: {r['train_loss_final']:.6f}")
        print(f"  - Val loss final:   {r['val_loss_final']:.6f}")
        print(f"  - Overfitting gap:  {r['overfit_gap']:.6f}")
        print(f"  - Test accuracy:    {r['test_accuracy']:.4f}")
        
        if r['overfit_gap'] > 0.01:
            print(f"  - Observation: OVERFITTING détecté (val loss > train loss)")
            print(f"  - Solution: Réduire la taille du modèle, ajouter L2, ou plus de données")
        elif r['overfit_gap'] < -0.001:
            print(f"  - Observation: Underfitting possible (train loss > val loss)")
            print(f"  - Solution: Augmenter la taille du modèle ou learning rate")
        else:
            print(f"  - Observation: Bon équilibre (train ≈ val)")
    
    # Export résultats
    print("\n[5/5] Exporting results...")
    
    export_dict = {
        'dataset': {
            'train_size': len(train_images),
            'val_size': len(val_images),
            'test_size': len(test_images),
            'num_classes': len(class_names)
        },
        'models': {}
    }
    
    for name, r in results.items():
        # Sauvegarder loss histories séparément
        loss_file = os.path.join(results_dir, f'etape_c_{name.lower()}_loss.csv')
        with open(loss_file, 'w') as f:
            f.write("epoch,train_loss,val_loss\n")
            for epoch, (tloss, vloss) in enumerate(zip(r['train_losses'], r['val_losses'])):
                f.write(f"{epoch+1},{tloss:.6f},{vloss:.6f}\n")
        
        # Ajouter au dict (sans les listes)
        export_dict['models'][name] = {
            'hidden_size': r['hidden_size'],
            'train_loss_final': r['train_loss_final'],
            'val_loss_final': r['val_loss_final'],
            'overfit_gap': r['overfit_gap'],
            'is_overfitting': r['is_overfitting'],
            'test_accuracy': r['test_accuracy'],
            'test_f1': r['test_f1']
        }
    
    metrics_file = os.path.join(results_dir, 'etape_c_overfitting_analysis.json')
    with open(metrics_file, 'w') as f:
        json.dump(export_dict, f, indent=2)
    
    print(f"✓ Analysis exported to {metrics_file}")
    
    print("\n" + "="*70)
    print("✓ ÉTAPE C Complete!")
    print("  Comparison summary saved")
    print("  Loss histories saved for plotting")
    print("  Next: Run ÉTAPE D for color & 5 classes")
    print("="*70)

if __name__ == "__main__":
    main()
