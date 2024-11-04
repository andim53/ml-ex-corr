#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Neural Network Implementation for Auto-MPG Dataset
Features:
- L2 regularization
- Dropout
- Batch normalization
- Improved architecture
- Better data normalization
- K-fold cross validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

@dataclass
class NetworkConfig:
    """Configuration for neural network hyperparameters."""
    layer_dims: List[int]
    activation_funcs: List[str]
    learning_rate: float = 0.0005  # Reduced learning rate
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_iterations: int = 500
    batch_size: int = 32
    early_stopping_patience: int = 20
    min_delta: float = 1e-4
    l2_lambda: float = 0.01  # L2 regularization parameter
    dropout_rate: float = 0.2  # Dropout rate
    use_batch_norm: bool = True

class ActivationFunctions:
    """Collection of activation functions and their derivatives."""
    
    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        return np.maximum(Z, 0)
    
    @staticmethod
    def relu_deriv(Z: np.ndarray) -> np.ndarray:
        return Z > 0
    
    @staticmethod
    def leaky_relu(Z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(Z > 0, Z, alpha * Z)
    
    @staticmethod
    def leaky_relu_deriv(Z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(Z > 0, 1, alpha)

class BatchNorm:
    """Batch Normalization layer."""
    def __init__(self, dim: int):
        self.gamma = np.ones((dim, 1))
        self.beta = np.zeros((dim, 1))
        self.epsilon = 1e-8
        self.moving_mean = np.zeros((dim, 1))
        self.moving_var = np.ones((dim, 1))
        self.momentum = 0.9

    def forward(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, dict]:
        if training:
            mu = np.mean(X, axis=1, keepdims=True)
            var = np.var(X, axis=1, keepdims=True)
            
            # Update moving averages
            self.moving_mean = (self.momentum * self.moving_mean + 
                              (1 - self.momentum) * mu)
            self.moving_var = (self.momentum * self.moving_var + 
                             (1 - self.momentum) * var)
        else:
            mu = self.moving_mean
            var = self.moving_var

        X_norm = (X - mu) / np.sqrt(var + self.epsilon)
        out = self.gamma * X_norm + self.beta
        
        cache = {'X_norm': X_norm, 'mu': mu, 'var': var, 'X': X}
        return out, cache

    def backward(self, dout: np.ndarray, cache: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_norm, mu, var, X = cache['X_norm'], cache['mu'], cache['var'], cache['X']
        m = X.shape[1]
        
        dgamma = np.sum(dout * X_norm, axis=1, keepdims=True)
        dbeta = np.sum(dout, axis=1, keepdims=True)
        
        dX_norm = dout * self.gamma
        dvar = np.sum(dX_norm * (X - mu) * -0.5 * (var + self.epsilon)**(-1.5), axis=1, keepdims=True)
        dmu = np.sum(dX_norm * -1/np.sqrt(var + self.epsilon), axis=1, keepdims=True)
        
        dX = (dX_norm / np.sqrt(var + self.epsilon) + 
              2 * dvar * (X - mu) / m + 
              dmu / m)
        
        return dX, dgamma, dbeta

class NeuralNetwork:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.params: Dict = {}
        self.batch_norm_layers = {}
        self.history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': [], 
            'train_mse': [], 'val_mse': []
        }
        self._initialize_parameters()
        
    def _initialize_parameters(self) -> None:
        """Initialize network parameters using He initialization."""
        for i in range(1, len(self.config.layer_dims)):
            # He initialization
            scale = np.sqrt(2.0 / self.config.layer_dims[i - 1])
            self.params[f'W{i}'] = np.random.randn(
                self.config.layer_dims[i], 
                self.config.layer_dims[i - 1]).astype(np.float32) * scale
            self.params[f'b{i}'] = np.zeros(
                (self.config.layer_dims[i], 1), 
                dtype=np.float32)
            self.params[f'activation{i}'] = self.config.activation_funcs[i - 1]
            
            # Initialize batch norm if enabled
            if self.config.use_batch_norm and i < len(self.config.layer_dims) - 1:
                self.batch_norm_layers[i] = BatchNorm(self.config.layer_dims[i])
        
        # Initialize Adam optimizer parameters
        for i in range(1, len(self.config.layer_dims)):
            for param in ['v_dW', 'v_db', 's_dW', 's_db']:
                if param not in self.params:
                    self.params[param] = {}
                self.params[param][f'd{"W" if "W" in param else "b"}{i}'] = \
                    np.zeros_like(self.params[f'{"W" if "W" in param else "b"}{i}'], 
                                dtype=np.float32)

    def _apply_dropout(self, A: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply dropout regularization."""
        if not training or self.config.dropout_rate == 0:
            return A, None
            
        mask = np.random.rand(*A.shape) > self.config.dropout_rate
        A *= mask / (1 - self.config.dropout_rate)  # Inverted dropout
        return A, mask

    def forward_propagation(self, X: np.ndarray, training: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
        """Perform forward propagation with dropout and batch normalization."""
        cache = {'dropout_masks': [], 'batch_norm': []}
        A = X
        Z_vals = []
        A_vals = [A]
        
        for i in range(1, len(self.config.layer_dims)):
            Z = self.params[f'W{i}'].dot(A) + self.params[f'b{i}']
            
            # Apply batch normalization before activation
            if self.config.use_batch_norm and i < len(self.config.layer_dims) - 1:
                Z, bn_cache = self.batch_norm_layers[i].forward(Z, training)
                cache['batch_norm'].append(bn_cache)
            else:
                cache['batch_norm'].append(None)
            
            Z_vals.append(Z)
            
            # Apply activation function
            if self.params[f'activation{i}'] == 'ReLu':
                A = ActivationFunctions.leaky_relu(Z)
            else:
                A = Z
            
            # Apply dropout
            if i < len(self.config.layer_dims) - 1:
                A, mask = self._apply_dropout(A, training)
                cache['dropout_masks'].append(mask)
            
            A_vals.append(A)
            
        return Z_vals, A_vals, cache

    def backward_propagation(self, Z_vals: List[np.ndarray], A_vals: List[np.ndarray], 
                           cache: Dict, X: np.ndarray, Y: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Perform backward propagation with regularization."""
        m = Y.shape[1]
        dW = {}
        db = {}
        
        # Output layer gradient
        dZ = np.sign(A_vals[-1] - Y)  # MAE gradient
        
        for layer in reversed(range(1, len(self.config.layer_dims))):
            # Add L2 regularization gradient
            reg_term = self.config.l2_lambda * self.params[f'W{layer}']
            
            if layer < len(self.config.layer_dims) - 1:
                # Apply dropout gradient
                if cache['dropout_masks'][layer-1] is not None:
                    dZ *= cache['dropout_masks'][layer-1] / (1 - self.config.dropout_rate)
            
            dW[layer] = (1 / m) * dZ.dot(A_vals[layer - 1].T) + reg_term
            db[layer] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            
            if layer > 1:
                dA = self.params[f'W{layer}'].T.dot(dZ)
                
                # Batch norm backward pass if enabled
                if self.config.use_batch_norm and cache['batch_norm'][layer-2] is not None:
                    dA, dgamma, dbeta = self.batch_norm_layers[layer-1].backward(dA, cache['batch_norm'][layer-2])
                
                # Activation function gradient
                if self.params[f'activation{layer-1}'] == 'ReLu':
                    dZ = dA * ActivationFunctions.leaky_relu_deriv(Z_vals[layer - 2])
                else:
                    dZ = dA
                    
        return dW, db

    def _compute_cost(self, AL: np.ndarray, Y: np.ndarray) -> float:
        """Compute cost with L2 regularization."""
        m = Y.shape[1]
        cost = np.mean(np.abs(AL - Y))  # MAE
        
        # Add L2 regularization term
        l2_cost = 0
        for i in range(1, len(self.config.layer_dims)):
            l2_cost += np.sum(np.square(self.params[f'W{i}']))
        cost += (self.config.l2_lambda / (2 * m)) * l2_cost
        
        return float(cost)

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, 
             X_val: np.ndarray, Y_val: np.ndarray) -> None:
        """Train the neural network with mini-batches and early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        m = X_train.shape[1]
        
        for i in range(1, self.config.max_iterations + 1):
            # Mini-batch training
            indices = np.random.permutation(m)
            num_batches = m // self.config.batch_size
            
            epoch_train_loss = 0
            for b in range(num_batches):
                batch_indices = indices[b*self.config.batch_size:(b+1)*self.config.batch_size]
                X_batch = X_train[:, batch_indices]
                Y_batch = Y_train[:, batch_indices]
                
                # Forward and backward passes
                Z_vals, A_vals, cache = self.forward_propagation(X_batch, training=True)
                dW, db = self.backward_propagation(Z_vals, A_vals, cache, X_batch, Y_batch)
                
                # Update parameters
                self._update_parameters(dW, db, i)
                
                epoch_train_loss += self._compute_cost(A_vals[-1], Y_batch)
            
            epoch_train_loss /= num_batches
            
            # Validation metrics
            Z_vals_val, A_vals_val, _ = self.forward_propagation(X_val, training=False)
            val_loss = self._compute_cost(A_vals_val[-1], Y_val)
            
            # Update history
            self.history['train_loss'].append(epoch_train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping triggered at iteration {i}")
                break
            
            if i % 10 == 0:
                print(f"Iteration {i}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {val_loss:.4f}")

    def _update_parameters(self, dW: Dict[int, np.ndarray], db: Dict[int, np.ndarray], t: int) -> None:
        """Update parameters using Adam optimizer with learning rate scheduling."""
        lr = self.config.learning_rate * (1.0 / (1.0 + 0.01 * t))
        
        for layer in range(1, len(self.config.layer_dims)):
            for param_name, grad, param_type in [('W', dW[layer], 'dW'), ('b', db[layer], 'db')]:
                # Momentum
                self.params[f'v_{param_type}'][f'{param_type}{layer}'] = \
                    (self.config.beta1 * self.params[f'v_{param_type}'][f'{param_type}{layer}'] + 
                     (1 - self.config.beta1) * grad)
                
                # RMSprop
                self.params[f's_{param_type}'][f'{param_type}{layer}'] = \
                    (self.config.beta2 * self.params[f's_{param_type}'][f'{param_type}{layer}'] + 
                     (1 - self.config.beta2) * (grad ** 2))
                
                # Bias correction
                v_corrected = self.params[f'v_{param_type}'][f'{param_type}{layer}'] / (1 - self.config.beta1 ** t)
                s_corrected = self.params[f's_{param_type}'][f'{param_type}{layer}'] / (1 - self.config.beta2 ** t)
                
                # Update parameters
                self.params[f'{param_name}{layer}'] -= \
                    (lr * v_corrected / (np.sqrt(s_corrected) + self.config.epsilon))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained network."""
        Z_vals, A_vals, _ = self.forward_propagation(X, training=False)
        return A_vals[-1]

    def plot_training_history(self) -> None:
        """Plot training metrics."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss', color='blue')
        plt.plot(self.history['val_loss'], label='Validation Loss', color='red')
        plt.title('Loss Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss (MAE)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def load_and_preprocess_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the auto-mpg dataset with robust scaling."""
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                   'Acceleration', 'Model Year', 'Origin']
    
    # Load data
    dataset = pd.read_csv(filepath, names=column_names, na_values='?', 
                         comment='\t', sep=' ', skipinitialspace=True)
    
    # Clean and preprocess
    dataset = dataset.dropna()
    
    # Convert categorical variables using one-hot encoding
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
    
    # Separate features and target
    X = dataset.drop('MPG', axis=1).values.T
    y = dataset['MPG'].values.reshape(1, -1)
    
    # Use robust scaling for features
    scaler = StandardScaler()
    X = scaler.fit_transform(X.T).T
    
    # Split data
    train_size = int(0.8 * X.shape[1])
    indices = np.random.permutation(X.shape[1])
    
    X_train = X[:, indices[:train_size]]
    Y_train = y[:, indices[:train_size]]
    X_val = X[:, indices[train_size:]]
    Y_val = y[:, indices[train_size:]]
    
    return X_train, Y_train, X_val, Y_val

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, config: NetworkConfig, 
                          n_splits: int = 5) -> Tuple[float, float]:
    """Perform k-fold cross validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scores = []
    
    # Transpose X for KFold
    X_t = X.T
    y_t = y.T
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_t)):
        X_train, X_val = X_t[train_idx].T, X_t[val_idx].T
        y_train, y_val = y_t[train_idx].T, y_t[val_idx].T
        
        model = NeuralNetwork(config)
        model.train(X_train, y_train, X_val, y_val)
        
        val_predictions = model.predict(X_val)
        mae = np.mean(np.abs(val_predictions - y_val))
        mae_scores.append(mae)
        print(f"Fold {fold + 1} MAE: {mae:.4f}")
    
    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)
    return mean_mae, std_mae

def main():
    # Load and preprocess data
    X_train, Y_train, X_val, Y_val = load_and_preprocess_data("dataset/auto-mpg.data")
    
    # Configure network with improved architecture
    config = NetworkConfig(
        layer_dims=[X_train.shape[0], 64, 32, 16, 1],  # Smaller network
        activation_funcs=['ReLu', 'ReLu', 'ReLu', 'identity'],
        learning_rate=0.0005,
        max_iterations=500,
        early_stopping_patience=20,
        l2_lambda=0.01,
        dropout_rate=0.2,
        use_batch_norm=True,
        batch_size=32
    )
    
    # Perform k-fold cross validation
    print("Performing 5-fold cross validation...")
    mean_mae, std_mae = k_fold_cross_validation(X_train, Y_train, config)
    print(f"\nCross-validation MAE: {mean_mae:.4f} ± {std_mae:.4f}")
    
    # Train final model
    print("\nTraining final model...")
    model = NeuralNetwork(config)
    model.train(X_train, Y_train, X_val, Y_val)
    
    # Evaluate model
    train_predictions = model.predict(X_train)
    train_mae = np.mean(np.abs(train_predictions - Y_train))
    print(f"Final Train MAE: {train_mae:.4f}")
    
    val_predictions = model.predict(X_val)
    val_mae = np.mean(np.abs(val_predictions - Y_val))
    print(f"Final Validation MAE: {val_mae:.4f}")
    
    # Plot training history
    model.plot_training_history()

if __name__ == "__main__":
    main()