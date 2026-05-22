# 🎭 Reconnaissance d'Émotions sur Smileys
## Réseau de Neurones From Scratch - Python Stdlib Only

Implémentation **100% from scratch** d'un système de reconnaissance d'émotions sur des images de smileys synthétiques. **Aucune librairie ML** (numpy, sklearn, tensorflow, keras, opencv) - **Python stdlib uniquement**.

---
la vidéo https://youtu.be/NI8TJqVHntE
## ✅ Objectifs Réalisés

- ✅ **Version 1**: 3 classes (Happy, Neutral, Sad) → **100% accuracy**
- ✅ **Version 2**: 5 classes (+ Angry, Surprised) → Implémenté
- ✅ **Dataset synthétique**: 8×8 grayscale + 16×16 RGB
- ✅ **Réseau from scratch**: Forward pass, backprop, loss, métriques manuelles
- ✅ **Validation rigoureuse**: Train/Val/Test stratifié (70/15/15)
- ✅ **Analyse overfitting**: 4 architectures comparées
- ✅ **Résultats**: Métriques JSON, courbes CSV, rapports HTML

---

## 🏗️ Architecture du Réseau

### Topologie

```
Entrée (d neurones)
    ↓
Dense [W₁, b₁] → ReLU (h neurones)
    ↓
Dense [W₂, b₂] → Softmax (C classes)
    ↓
Sortie (probabilités)
```

### Dimensions par Étape

| Étape | Entrée | Caché | Sortie | Params |
|-------|--------|-------|--------|--------|
| **A/B** | 64 (8×8) | 8 | 3 | 520 |
| **C-Small** | 64 | 4 | 3 | 260 |
| **C-Medium** | 64 | 8 | 3 | 520 |
| **C-Large** | 64 | 32 | 3 | 2080 |
| **C-XL** | 64 | 64 | 3 | 4160 |
| **D** | 768 (16×16×3) | 16 | 5 | 12365 |

### Équations Mathématiques

**Forward Pass:**
```
z₁ = W₁ · x + b₁
a₁ = ReLU(z₁) = max(0, z₁)
z₂ = W₂ · a₁ + b₂
ŷ = softmax(z₂)
```

**Loss (Cross-Entropy):**
```
L = -Σᵢ yᵢ · log(ŷᵢ)
```

**Update (SGD):**
```
θ ← θ - α · ∇L
```

### Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| Learning Rate | 0.1 (0.05 XL) |
| Optimiseur | SGD simple |
| Epochs | 50-100 |
| Batch Size | 1 (online learning) |
| Loss | Cross-Entropy |
| Init Poids | Xavier Uniform |
| Init Biais | Zéros |

---

## 📁 Structure du Projet

```
projectspéIA/
├── src/                           Code source (10 fichiers Python)
│   ├── generate_data.py           Génère images B&W 8×8 (PGM)
│   ├── generate_data_color.py     Génère images couleur 16×16 (PPM)
│   ├── image_loader.py            Charge PGM/PPM, normalise
│   ├── model.py                   Réseau de neurones 2-couches
│   ├── metrics.py                 Confusion matrix, F1, accuracy
│   ├── validation.py              Stratified split train/val/test
│   ├── train.py                   Étape A (Baseline)
│   ├── train_etape_b.py           Étape B (Validation propre)
│   ├── train_etape_c.py           Étape C (Overfitting analysis)
│   └── train_etape_d.py           Étape D (Couleur + 5 classes)
├── data/                          Données synthétiques (229 images)
│   ├── train/                     75 images B&W + 100 images couleur
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   ├── happy_color/
│   │   ├── neutral_color/
│   │   ├── sad_color/
│   │   ├── angry_color/
│   │   └── surprised_color/
│   └── test/                      24 images B&W + 30 images couleur
├── results/                       Résultats et métriques
│   ├── baseline_metrics.json      Métriques Étape A
│   ├── etape_b_metrics.json       Métriques Étape B
│   ├── etape_c_overfitting_analysis.json
│   ├── etape_d_color_analysis.json
│   ├── *.csv                      Courbes de perte
│   ├── rapport.html               Rapport interactif
│   └── rapport_complet.html       Rapport 5-10 pages
└── README.md                      Documentation
```

---

## 🚀 Démarrage Rapide

### Prérequis

- **Python 3.6+**
- **Aucune dépendance externe** (stdlib uniquement)

### Installation

```bash
cd /Users/laly/Sites/A4/spéIA/projectspéIA
# C'est tout! Aucune installation nécessaire
```

### Lancer les Expériences

#### Étape A: Baseline Simple
```bash
python3 src/train.py
```
- Génère dataset 8×8 B&W (3 classes)
- Entraîne réseau 64→8→3
- **Résultat:** 100% accuracy
- **Fichier:** `results/baseline_metrics.json`

#### Étape B: Validation Propre
```bash
python3 src/train_etape_b.py
```
- Split stratifié 70/15/15 (train/val/test)
- Validation monitoring + early stopping
- **Résultat:** 100% test accuracy
- **Fichier:** `results/etape_b_metrics.json`

#### Étape C: Overfitting Analysis
```bash
python3 src/train_etape_c.py
```
- Compare 4 architectures (4, 8, 32, 64 neurones)
- Affiche train/val loss curves
- **Résultat:** Pas d'overfitting détecté
- **Fichiers:** `results/etape_c_*.json` + `results/etape_c_*_loss.csv`

#### Étape D: Couleur + 5 Classes
```bash
python3 src/train_etape_d.py
```
- Génère dataset couleur PPM 16×16×RGB (5 classes)
- Entraîne modèle 768→16→5
- Compares B&W vs Couleur
- **Résultat:** B&W 100%, Couleur 20%
- **Fichier:** `results/etape_d_color_analysis.json`

#### Lancer Tout d'Un Coup

```bash
python3 src/train.py && \
python3 src/train_etape_b.py && \
python3 src/train_etape_c.py && \
python3 src/train_etape_d.py && \
echo "✅ Tous les résultats sont prêts!"
```

---

## 📊 Résultats

### Étapes A & B: Baseline (3 Classes B&W - 8×8)

| Métrique | Étape A | Étape B |
|----------|---------|---------|
| Train Accuracy | 100% | 100% |
| **Test Accuracy** | **100%** | **100%** |
| Macro F1 | 1.0000 | 1.0000 |
| Train Loss | 0.0000 | 0.00130 |
| Val Loss | - | 0.00120 |

**Confusion Matrix (Test):**
```
         Happy  Neutral  Sad
Happy       8        0      0
Neutral     0        8      0
Sad         0        0      8
```

### Étape C: Overfitting Analysis

| Modèle | Hidden | Train Loss | Val Loss | Test Acc | Overfit? |
|--------|--------|-----------|----------|----------|----------|
| Small  | 4 | 0.000662 | 0.000517 | 100% | ❌ NO |
| Medium | 8 | 0.001298 | 0.001201 | 100% | ❌ NO |
| Large | 32 | 0.000278 | 0.000237 | 100% | ❌ NO |
| XL | 64 | 0.000624 | 0.000506 | 100% | ❌ NO |

**Conclusion:** ✅ **Aucun overfitting détecté.** Dataset trop simple (séparation linéaire facile).

### Étape D: Couleur + 5 Classes

| Aspect | B&W 3-class | Couleur 5-class |
|--------|------------|-----------------|
| Résolution | 8×8 | 16×16 |
| Pixels | 64 | 768 (12×) |
| Classes | 3 | 5 |
| **Test Accuracy** | **100%** | **20%** |

**Analyse:** Performance réduite sur couleur car 5 classes et 12× plus de pixels. Besoin d'architecture plus grande ou plus d'epochs.

---

## 🔍 Détails Implémentation

### Modules Core

**model.py:**
- Classe `NeuralNetwork` complète
- Forward pass: z1, a1, z2, output
- Backward pass: gradients complets
- Loss: cross-entropy avec clipping numérique
- Update: SGD simple

**generate_data.py:**
- Crée images PGM 8×8 synthétiques
- 3 émotions (Happy, Neutral, Sad)
- Variation aléatoire pour augmentation

**generate_data_color.py:**
- Crée images PPM 16×16×RGB
- 5 émotions avec couleurs distinctes
- Happy (rouge), Neutral (gris), Sad (bleu), Angry (rouge), Surprised (jaune)

**image_loader.py:**
- Parsing manuel PGM/PPM (pas d'opencv)
- Normalisation [0,255] → [0,1]
- Conversion images → vecteurs

**validation.py:**
- Stratified split (préserve distribution classes)
- Train/Val/Test split (70/15/15)

**metrics.py:**
- Confusion matrix manuelle
- Accuracy, Precision, Recall, F1
- Macro F1 (moyenne non-pondérée)

### Stabilité Numérique

✅ **Softmax Safe:**
```python
z_max = max(z)
exp_z = [exp(zi - z_max) for zi in z]
return [e / sum(exp_z) for e in exp_z]
```

✅ **Cross-Entropy Safe:**
```python
pred = clip(pred, 1e-10, 1-1e-10)
loss = -sum(y * log(pred))
```

---

## ⚠️ Contraintes Respectées

| Élément | Autorisé | Interdit |
|---------|----------|----------|
| **Stdlib Python** | ✅ math, random, json, csv, os | - |
| **Numpy** | ❌ | ✅ Interdite |
| **ML Frameworks** | ❌ | ✅ sklearn, tensorflow, keras, pytorch |
| **Image Processing** | ❌ | ✅ opencv, PIL, matplotlib |
| **Formats** | ✅ Parsing manuel | - |
| **Opérations** | ✅ Manuelles (listes/floats) | - |

**Résultat:** 100% from scratch - chaque ligne expliquable

---

## 📝 Format des Données

### Images PGM (Noir & Blanc)

```
P2
8 8
255
[64 pixel values 0-255]
```

- Format ASCII texte
- Header P2 (magic number)
- Dimension et max value
- Pixels grayscale [0-255]

### Images PPM (Couleur)

```
P3
16 16
255
[768 RGB values = 256 triplets]
```

- Format ASCII texte
- Header P3 (magic number)
- Dimension et max value
- RGB triplets [0-255] pour chaque pixel

---

## 📈 Fichiers Résultats

### Métriques JSON
- `results/baseline_metrics.json` - Étape A
- `results/etape_b_metrics.json` - Étape B
- `results/etape_c_overfitting_analysis.json` - Analyse overfitting
- `results/etape_d_color_analysis.json` - Comparaison couleur

### Courbes de Perte (CSV)
- `results/training_loss.csv` - Courbes Étape A
- `results/etape_b_loss.csv` - Courbes Étape B
- `results/etape_c_*_loss.csv` - Courbes Étape C (4 fichiers)



---

## 📄 Consulter les Résultats

### Afficher les métriques
```bash
cat results/baseline_metrics.json
cat results/etape_b_metrics.json
```

### Ouvrir le rapport interactif
```bash
open results/rapport.html
```

### Ouvrir le rapport complet
```bash
open results/rapport_complet.html
# Puis: Cmd+P → Save as PDF
```

---

## 🎓 Concepts Expliqués

### Architecture
- **Dense layers:** Chaque neurone connecté à tous les inputs
- **ReLU:** max(0, x) - activation non-linéaire
- **Softmax:** Convertit scores en probabilités (Σ=1)

### Optimisation
- **SGD:** Stochastic Gradient Descent
- **Backpropagation:** Chaîne de dérivées pour gradients
- **Learning rate:** Taille des pas de mise à jour

### Validation
- **Train/Val/Test split:** 70% entraînement, 15% validation, 15% test
- **Stratified split:** Préserve distribution classes
- **Early stopping:** Arrête si validation loss stagne

### Overfitting
- **Définition:** Modèle apprend train parfaitement mais échoue sur test
- **Signes:** Train loss << Val loss
- **Solutions:** Modèle plus petit, plus de données, L2 regularization

---

## 🚀 Pistes d'Amélioration

### Court terme
- Augmenter epochs pour modèle couleur (200+)
- Agrandir couche cachée (64+ neurones pour couleur)
- Ajouter données synthétiques bruitées

### Moyen terme
- Régularisation L1/L2
- Batch gradient descent
- Learning rate adaptatif (momentum, Adam)
- Batch normalization

### Long terme
- Architectures profondes (3+ couches)
- Convolutional networks (CNN)
- Transfer learning
- Datasets réels (FER-2013, CK+)

---

## ✅ Checklist Livraison

- ✅ Code source complet (10 fichiers Python)
- ✅ Données synthétiques (229 images)
- ✅ Résultats chiffrés (JSON, CSV)
- ✅ Rapports HTML (interactif + complet)
- ✅ Documentation (README.md)
- ✅ Aucune dépendance externe
- ✅ Contraintes respectées
- ✅ 100% accuracy sur 3 classes B&W
- ✅ Analyse complète overfitting
- ✅ Extension 5 classes couleur

**Status:** ✅ **PRÊT POUR REMISE** (19 mai 2026)

---

## 📞 Support

Pour questions:
1. Vérifier que Python 3.6+ est installé
2. Vérifier que toutes les données sont dans `data/`
3. Relancer le script depuis le bon répertoire
4. Consulter les logs dans le terminal

---

