"""
metrics.py - Calcul des métriques de classification.
No external libraries: ONLY Python standard.
"""

def confusion_matrix(y_true, y_pred, num_classes):
    """
    Construit la matrice de confusion.
    
    Args:
        y_true: liste des vrais labels (indices)
        y_pred: liste des prédictions (indices)
        num_classes: nombre de classes
    
    Retour:
        Matrice de confusion (liste de listes)
        matrix[i][j] = nombre de fois où la vraie classe est i et prédite est j
    """
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1
    
    return matrix

def accuracy(y_true, y_pred):
    """
    Accuracy = nombre de prédictions correctes / nombre total
    """
    if not y_true:
        return 0.0
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def precision_recall_f1(matrix, class_id):
    """
    Calcule precision, recall et F1 pour une classe donnée.
    
    Args:
        matrix: matrice de confusion
        class_id: indice de la classe
    
    Retour:
        (precision, recall, f1)
    
    Formules:
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * (precision * recall) / (precision + recall)
    """
    # TP: diagonal de la matrice
    tp = matrix[class_id][class_id]
    
    # FP: somme de la colonne sauf la diagonale
    fp = sum(matrix[r][class_id] for r in range(len(matrix)) if r != class_id)
    
    # FN: somme de la ligne sauf la diagonale
    fn = sum(matrix[class_id][c] for c in range(len(matrix)) if c != class_id)
    
    # Calculs
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def macro_f1(matrix, class_names):
    """
    Calcule la macro F1: moyenne des F1 par classe.
    
    Args:
        matrix: matrice de confusion
        class_names: liste des noms de classes
    
    Retour:
        macro_f1 value
    """
    num_classes = len(class_names)
    f1_scores = []
    
    for class_id in range(num_classes):
        _, _, f1 = precision_recall_f1(matrix, class_id)
        f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

def print_metrics(y_true, y_pred, class_names):
    """
    Affiche tous les métriques de classification.
    
    Args:
        y_true: vraies labels
        y_pred: prédictions
        class_names: liste des noms de classes
    """
    num_classes = len(class_names)
    
    # Matrice de confusion
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    
    # Accuracy
    acc = accuracy(y_true, y_pred)
    
    # Métriques par classe
    print("\n" + "="*60)
    print(f"{'Classe':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*60)
    
    f1_scores = []
    for class_id, class_name in enumerate(class_names):
        prec, rec, f1 = precision_recall_f1(matrix, class_id)
        print(f"{class_name:<15} {prec:.4f}        {rec:.4f}        {f1:.4f}")
        f1_scores.append(f1)
    
    # Macro F1
    macro_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    print("-"*60)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1_score:.4f}")
    print("="*60)
    
    # Matrice de confusion détaillée
    print("\nConfusion Matrix:")
    print("Rows = True labels, Columns = Predicted labels\n")
    
    # Header
    print(f"{'True\\Pred':<10}", end="")
    for class_name in class_names:
        print(f"{class_name:<10}", end="")
    print()
    
    # Rows
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<10}", end="")
        for j in range(num_classes):
            print(f"{matrix[i][j]:<10}", end="")
        print()
    
    print()
    return matrix, acc, macro_f1_score

def export_metrics_json(y_true, y_pred, class_names, filepath):
    """
    Exporte les métriques en JSON.
    """
    import json
    
    num_classes = len(class_names)
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    acc = accuracy(y_true, y_pred)
    
    metrics_dict = {
        'accuracy': acc,
        'confusion_matrix': matrix,
        'per_class_metrics': {}
    }
    
    f1_scores = []
    for class_id, class_name in enumerate(class_names):
        prec, rec, f1 = precision_recall_f1(matrix, class_id)
        metrics_dict['per_class_metrics'][class_name] = {
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
        f1_scores.append(f1)
    
    metrics_dict['macro_f1'] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"✓ Metrics exported to {filepath}")
