
import math
import random

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        """
        Initialise le réseau.
        
        Args:
            input_size: dimension d'entrée (ex: 64 pour 8×8)
            hidden_size: nombre de neurones cachés
            output_size: nombre de classes (3 ou 5)
            seed: pour reproductibilité
        """
        random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialisation des poids et biais 
        init_limit_1 = math.sqrt(6.0 / (input_size + hidden_size))
        init_limit_2 = math.sqrt(6.0 / (hidden_size + output_size))
        
        # Couche 1: input -> hidden
        self.w1 = [[random.uniform(-init_limit_1, init_limit_1) for _ in range(input_size)]
                   for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        
        # Couche 2: hidden -> output
        self.w2 = [[random.uniform(-init_limit_2, init_limit_2) for _ in range(hidden_size)]
                   for _ in range(output_size)]
        self.b2 = [0.0] * output_size
        
        # Régularisation L2
        self.l2_lambda = 0.0
    
    def _relu(self, z):
        """ReLU activation: max(0, z)"""
        return max(0.0, z)
    
    def _relu_derivative(self, z):
        """Dérivée de ReLU: 1 si z > 0, else 0"""
        return 1.0 if z > 0.0 else 0.0
    
    def _softmax(self, z_list):
        """
        Softmax: softmax(z)_k = exp(z_k) / sum(exp(z))
        Retourne une liste de probabilités.
        """
        # Soustrait le max pour éviter overflow
        max_z = max(z_list)
        exp_z = [math.exp(z - max_z) for z in z_list]
        sum_exp = sum(exp_z)
        return [e / sum_exp for e in exp_z]
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: vecteur d'entrée (liste de floats)
        
        Retour:
            (z1, a1, z2, output, cache)
            - z1: pre-activation couche 1
            - a1: activation couche 1 (ReLU)
            - z2: pre-activation couche 2
            - output: softmax (probabilités)
            - cache: toutes les valeurs utiles pour backprop
        """
        # Couche 1: z1 = w1 @ x + b1
        z1 = []
        for j in range(self.hidden_size):
            z1_j = self.b1[j]
            for i in range(self.input_size):
                z1_j += self.w1[j][i] * x[i]
            z1.append(z1_j)
        
        # Activation 1: a1 = ReLU(z1)
        a1 = [self._relu(z) for z in z1]
        
        # Couche 2: z2 = w2 @ a1 + b2
        z2 = []
        for k in range(self.output_size):
            z2_k = self.b2[k]
            for j in range(self.hidden_size):
                z2_k += self.w2[k][j] * a1[j]
            z2.append(z2_k)
        
        # Activation 2: softmax
        output = self._softmax(z2)
        
        cache = {
            'x': x,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'output': output
        }
        
        return z1, a1, z2, output, cache
    
    def _cross_entropy_loss(self, y_true, y_pred):
        """
        Cross-entropy loss: L = -sum(y_true * log(y_pred))
        
        Args:
            y_true: one-hot encoded label (liste de 0/1)
            y_pred: softmax output (probabilités)
        
        Retour: loss value (float)
        """
        loss = 0.0
        for i in range(self.output_size):
            if y_true[i] > 0:  # y_true[i] == 1
                # Clamp pour éviter log(0)
                pred = max(1e-10, y_pred[i])
                loss -= math.log(pred)
        return loss
    
    def backward(self, y_true, cache):
        """
        Backpropagation: calcule les gradients.
        
        Args:
            y_true: one-hot encoded label
            cache: résultats du forward
        
        Retour:
            (grad_w1, grad_b1, grad_w2, grad_b2)
        """
        x = cache['x']
        z1 = cache['z1']
        a1 = cache['a1']
        output = cache['output']
        
        # Gradient de la sortie (softmax + cross-entropy se simplifient)
        delta2 = [output[k] - y_true[k] for k in range(self.output_size)]
        
        # Gradients couche 2
        grad_w2 = [[0.0] * self.hidden_size for _ in range(self.output_size)]
        grad_b2 = [0.0] * self.output_size
        
        for k in range(self.output_size):
            grad_b2[k] = delta2[k]
            for j in range(self.hidden_size):
                grad_w2[k][j] = a1[j] * delta2[k]
        
        # Backprop vers la couche cachée
        delta1 = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            sum_delta = 0.0
            for k in range(self.output_size):
                sum_delta += self.w2[k][j] * delta2[k]
            delta1[j] = sum_delta * self._relu_derivative(z1[j])
        
        # Gradients couche 1
        grad_w1 = [[0.0] * self.input_size for _ in range(self.hidden_size)]
        grad_b1 = [0.0] * self.hidden_size
        
        for j in range(self.hidden_size):
            grad_b1[j] = delta1[j]
            for i in range(self.input_size):
                grad_w1[j][i] = x[i] * delta1[j]
        
        return grad_w1, grad_b1, grad_w2, grad_b2
    
    def update(self, grad_w1, grad_b1, grad_w2, grad_b2, learning_rate):
        """
        Update les poids avec SGD: theta <- theta - lr * grad
        (Inclut L2 régularisation si activée)
        
        Args:
            grad_w1, grad_b1, grad_w2, grad_b2: gradients
            learning_rate: taux d'apprentissage
        """
        # Couche 1
        for j in range(self.hidden_size):
            for i in range(self.input_size):
                reg_term = 2 * self.l2_lambda * self.w1[j][i] if self.l2_lambda > 0 else 0
                self.w1[j][i] -= learning_rate * (grad_w1[j][i] + reg_term)
            self.b1[j] -= learning_rate * grad_b1[j]
        
        # Couche 2
        for k in range(self.output_size):
            for j in range(self.hidden_size):
                reg_term = 2 * self.l2_lambda * self.w2[k][j] if self.l2_lambda > 0 else 0
                self.w2[k][j] -= learning_rate * (grad_w2[k][j] + reg_term)
            self.b2[k] -= learning_rate * grad_b2[k]
    
    def predict(self, x):
        """
        Prédiction: retourne la classe avec la probabilité maximale.
        
        Args:
            x: vecteur d'entrée
        
        Retour:
            class_id (0, 1, 2, ...)
        """
        _, _, _, output, _ = self.forward(x)
        return output.index(max(output))
    
    def compute_loss(self, images, labels, regularization=False):
        """
        Calcule la loss moyenne sur un batch.
        
        Args:
            images: liste de vecteurs
            labels: liste d'indices
            regularization: inclure L2 dans la loss?
        
        Retour: loss moyenne
        """
        total_loss = 0.0
        for x, label in zip(images, labels):
            # One-hot encoding
            y_true = [0.0] * self.output_size
            y_true[label] = 1.0
            
            _, _, _, output, _ = self.forward(x)
            total_loss += self._cross_entropy_loss(y_true, output)
        
        avg_loss = total_loss / len(images) if images else 0.0
        
        # Ajouter L2 regularization term
        if regularization and self.l2_lambda > 0:
            l2_term = 0.0
            for j in range(self.hidden_size):
                for i in range(self.input_size):
                    l2_term += self.w1[j][i] ** 2
            for k in range(self.output_size):
                for j in range(self.hidden_size):
                    l2_term += self.w2[k][j] ** 2
            avg_loss += self.l2_lambda * l2_term
        
        return avg_loss
    
    def train_one_epoch(self, images, labels, learning_rate):
        """
        Une époque d'entraînement (passe sur tous les exemples).
        
        Args:
            images: liste de vecteurs d'entrée
            labels: liste d'indices de classe
            learning_rate: taux d'apprentissage
        
        Retour: loss moyenne de l'époque
        """
        total_loss = 0.0
        
        for x, label in zip(images, labels):
            # One-hot encoding
            y_true = [0.0] * self.output_size
            y_true[label] = 1.0
            
            # Forward
            _, _, _, output, cache = self.forward(x)
            loss = self._cross_entropy_loss(y_true, output)
            total_loss += loss
            
            # Backward
            grad_w1, grad_b1, grad_w2, grad_b2 = self.backward(y_true, cache)
            
            # Update
            self.update(grad_w1, grad_b1, grad_w2, grad_b2, learning_rate)
        
        return total_loss / len(images) if images else 0.0
