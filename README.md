# MAYINI Deep Learning Framework

[![PyPI version](https://badge.fury.io/py/mayini-framework.svg)](https://badge.fury.io/py/mayini-framework)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/907-bot-collab/mayini/workflows/CI/badge.svg)](https://github.com/907-bot-collab/mayini/actions)

MAYINI is a comprehensive deep learning framework built from scratch in Python, featuring automatic differentiation, neural network components, and complete training infrastructure. It's designed for educational purposes and research, providing a PyTorch-like API with full transparency into the underlying mechanics.

## 🚀 Key Features

- **Complete Tensor Engine** with automatic differentiation
- **Neural Network Layers**: Linear, Conv2D, Pooling, BatchNorm, Dropout
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, GELU, LeakyReLU
- **RNN Components**: Vanilla RNN, LSTM, GRU with multi-layer support
- **Loss Functions**: MSE, MAE, CrossEntropy, BCE, Huber
- **Optimizers**: SGD, Adam, AdamW, RMSprop
- **Learning Rate Schedulers**: StepLR, ExponentialLR, CosineAnnealingLR
- **Training Infrastructure**: DataLoader, Trainer, Metrics, Early Stopping
- **Educational Focus**: Clear implementations with mathematical formulas

## 📦 Installation

```bash
pip install mayini-framework
```

## 🎓 Try It Now

**Interactive Colab Notebook**: [Open in Google Colab](https://colab.research.google.com/drive/140HDqQ3vBGy6HIpzbvH8Jv54PeLylNOK?usp=sharing)

The notebook contains 38 working examples demonstrating all framework features!

---

## 📚 Quick Start Guide

### 1. Tensor Operations with Autograd

```python
import mayini as mn
import numpy as np

# Create tensors with gradient tracking
x = mn.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = mn.Tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)

# Perform operations
z = x.matmul(y)      # Matrix multiplication
w = x + y            # Element-wise addition
loss = z.sum()

# Automatic differentiation
loss.backward()
print(f"Gradient of x: {x.grad}")
# Output: [[3. 3.] [3. 3.]]
```

### 2. Building Neural Networks

```python
from mayini.nn import Sequential, Linear, ReLU, Softmax

model = Sequential(
    Linear(784, 256, init_method='he'),
    ReLU(),
    Linear(256, 128, init_method='he'),
    ReLU(),
    Linear(128, 10),
    Softmax(dim=1)
)

# Forward pass
x = mn.Tensor(np.random.randn(32, 784))
output = model(x)
print(f"Output shape: {output.shape}")  # (32, 10)
```

### 3. Complete Training Example

```python
from mayini.nn import CrossEntropyLoss
from mayini.optim import Adam
from mayini.training import DataLoader, Trainer

# Prepare data
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)
X_val = np.random.randn(200, 784).astype(np.float32)
y_val = np.random.randint(0, 10, 200)

train_loader = DataLoader(X_train, y_train, batch_size=64, shuffle=True)
val_loader = DataLoader(X_val, y_val, batch_size=64, shuffle=False)

# Setup training
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()
trainer = Trainer(model, optimizer, criterion)

# Train
history = trainer.fit(
    train_loader,
    epochs=10,
    val_loader=val_loader,
    verbose=True
)

print(f"Final training accuracy: {history['train_acc'][-1]:.4f}")
print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
```

---

## 📖 Complete API Reference

### Core Components

#### Tensor
Core tensor class with automatic differentiation.

**Key Methods:**
- `matmul(other)` - Matrix multiplication
- `sum(axis=None, keepdims=False)` - Sum reduction
- `mean(axis=None, keepdims=False)` - Mean reduction
- `reshape(shape)` - Reshape tensor
- `transpose(axes=None)` - Transpose dimensions
- `backward(gradient=None)` - Compute gradients
- `zero_grad()` - Reset gradients

```python
# Example
x = mn.Tensor([[1, 2], [3, 4]], requires_grad=True)
y = x.matmul(x.transpose())
y.sum().backward()
```

---

### Neural Network Layers

#### Linear (Fully Connected)
```python
from mayini.nn import Linear

layer = Linear(
    in_features=784,
    out_features=256,
    bias=True,
    init_method='xavier'  # 'xavier', 'he', or 'normal'
)
```

#### Conv2D (2D Convolution)
```python
from mayini.nn import Conv2D

conv = Conv2D(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True
)
```

#### Pooling Layers
```python
from mayini.nn import MaxPool2D, AvgPool2D

max_pool = MaxPool2D(kernel_size=2, stride=2, padding=0)
avg_pool = AvgPool2D(kernel_size=2, stride=2, padding=0)
```

#### Batch Normalization
```python
from mayini.nn import BatchNorm1d

bn = BatchNorm1d(num_features=256, eps=1e-5, momentum=0.1)
```

#### Dropout
```python
from mayini.nn import Dropout

dropout = Dropout(p=0.5)
dropout.train()  # Enable dropout
dropout.eval()   # Disable dropout
```

#### Flatten
```python
from mayini.nn import Flatten

flatten = Flatten(start_dim=1)
```

---

### Activation Functions

All activation functions with mathematical formulas and use cases:

#### ReLU
**Formula:** f(x) = max(0, x)  
**Use case:** Most common for hidden layers

```python
from mayini.nn import ReLU
relu = ReLU()
```

#### Sigmoid
**Formula:** f(x) = 1 / (1 + e^(-x))  
**Use case:** Binary classification, LSTM gates

```python
from mayini.nn import Sigmoid
sigmoid = Sigmoid()
```

#### Tanh
**Formula:** f(x) = (e^x - e^(-x)) / (e^x + e^(-x))  
**Use case:** RNNs, zero-centered activation

```python
from mayini.nn import Tanh
tanh = Tanh()
```

#### Softmax
**Formula:** f(x_i) = e^(x_i) / Σ e^(x_j)  
**Use case:** Multi-class classification output

```python
from mayini.nn import Softmax
softmax = Softmax(dim=1)
```

#### GELU
**Use case:** Transformers, BERT, GPT models

```python
from mayini.nn import GELU
gelu = GELU()
```

#### Leaky ReLU
**Formula:** f(x) = max(αx, x) where α = 0.01  
**Use case:** Prevent dead neurons

```python
from mayini.nn import LeakyReLU
leaky_relu = LeakyReLU(negative_slope=0.01)
```

---

### Recurrent Neural Networks

#### RNN Cell
```python
from mayini.nn import RNNCell

rnn_cell = RNNCell(input_size=100, hidden_size=128, bias=True)
h_next = rnn_cell(x_t, h_t)
```

#### LSTM Cell
**Gates:** Forget, Input, Output, Cell Candidate  
**Formula:**
- Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
- Input gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
- Output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
- Cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_C · [h_{t-1}, x_t])

```python
from mayini.nn import LSTMCell

lstm_cell = LSTMCell(input_size=100, hidden_size=128, bias=True)

# Single timestep
x_t = mn.Tensor(np.random.randn(32, 100))
h_t = mn.Tensor(np.random.randn(32, 128))
c_t = mn.Tensor(np.random.randn(32, 128))

# ✅ FIX: Call .forward() directly
h_next, c_next = lstm_cell.forward(x_t, (h_t, c_t))
print(f"Next hidden: {h_next.shape}, Next cell: {c_next.shape}")

```

#### GRU Cell
**Gates:** Reset, Update, New  
**Formula:**
- Reset gate: r_t = σ(W_r · [h_{t-1}, x_t])
- Update gate: z_t = σ(W_z · [h_{t-1}, x_t])
- Hidden state: h_t = (1 - z_t) ⊙ tanh(W · [r_t ⊙ h_{t-1}, x_t]) + z_t ⊙ h_{t-1}

```python
from mayini.nn import GRUCell

gru_cell = GRUCell(input_size=100, hidden_size=128, bias=True)

# Single timestep
x_t = mn.Tensor(np.random.randn(32, 100))
h_t = mn.Tensor(np.random.randn(32, 128))

# ✅ FIX: Call .forward() directly
h_next = gru_cell.forward(x_t, h_t)
print(f"Next hidden state: {h_next.shape}")

```

#### Multi-layer RNN
```python
from mayini.nn import RNN

# Multi-layer LSTM
lstm_model = RNN(
    input_size=100,
    hidden_size=128,
    num_layers=2,
    cell_type='lstm',
    dropout=0.2,
    batch_first=True
)

# Process sequences
x_seq = mn.Tensor(np.random.randn(32, 50, 100))  # (batch, seq_len, features)

# ✅ FIX: This will work after you fix Module.__call__() in modules.py
# OR use this temporary workaround:
output, hidden_states = lstm_model.forward(x_seq)

print(f"Output shape: {output.shape}")
print(f"Number of hidden states: {len(hidden_states)}")

```

---

### Loss Functions

#### MSE Loss
**Formula:** L = (1/n) Σ (y_i - ŷ_i)²  
**Use case:** Regression tasks

```python
from mayini.nn import MSELoss
criterion = MSELoss(reduction='mean')  # 'mean', 'sum', or 'none'
```

#### MAE Loss
**Formula:** L = (1/n) Σ |y_i - ŷ_i|  
**Use case:** Robust regression

```python
from mayini.nn import MAELoss
criterion = MAELoss(reduction='mean')
```

#### Cross-Entropy Loss
**Formula:** L = -(1/n) Σ log(e^(f_yi) / Σ e^(f_j))  
**Use case:** Multi-class classification

```python
from mayini.nn import CrossEntropyLoss

criterion = CrossEntropyLoss(reduction='mean')
```

#### Binary Cross-Entropy
**Formula:** L = -(1/n) Σ [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]  
**Use case:** Binary classification

```python
from mayini.nn import BCELoss
criterion = BCELoss(reduction='mean')
```

#### Huber Loss
**Use case:** Robust regression with outliers

```python
from mayini.nn import HuberLoss
criterion = HuberLoss(delta=1.0, reduction='mean')
```

---

### Optimizers

#### SGD (Stochastic Gradient Descent)
**Update rule:** v_t = β·v_{t-1} + g_t, θ_t = θ_{t-1} - η·v_t

```python
from mayini.optim import SGD

optimizer = SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)
```

#### Adam
**Update rule:** Adaptive moment estimation with bias correction

```python
from mayini.optim import Adam

optimizer = Adam(
    model.parameters(),
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0
)
```

#### AdamW
**Feature:** Decoupled weight decay

```python
from mayini.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

#### RMSprop
```python
from mayini.optim import RMSprop

optimizer = RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,
    momentum=0.0
)
```

---

### Learning Rate Schedulers

#### StepLR
Decays LR by gamma every step_size epochs

```python
from mayini.optim import StepLR

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(50):
    train_one_epoch()
    scheduler.step()
```

#### ExponentialLR
Exponential decay by gamma each epoch

```python
from mayini.optim import ExponentialLR

scheduler = ExponentialLR(optimizer, gamma=0.95)
```

#### CosineAnnealingLR
Cosine annealing schedule

```python
from mayini.optim import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
```

---

### Training Utilities

#### DataLoader
```python
from mayini.training import DataLoader

train_loader = DataLoader(
    X_train,
    y_train,
    batch_size=64,
    shuffle=True
)

for batch_X, batch_y in train_loader:
    # Training code
    pass
```

#### Trainer
```python
from mayini.training import Trainer

trainer = Trainer(
    model,      # Neural network model (Module)
    optimizer,  # Optimization algorithm (Optimizer)
    criterion   # Loss function (Module)
)

```

**Trainer Methods:**
- `fit()` - Train the model
- `evaluate()` - Evaluate on test data
- `predict()` - Make predictions
- `save_checkpoint()` - Save model state
- `load_checkpoint()` - Load model state
  
####fit()
```python
history = trainer.fit(
    train_loader,              # Training data loader
    epochs=10,                 # Number of training epochs
    val_loader=None,           # Optional validation data loader
    early_stopping=None,       # Optional early stopping callback
    verbose=True,              # Print training progress
    save_best=True,            # Save best model based on validation loss
    checkpoint_path='model.pkl' # Path to save checkpoints
)
```
#### Metrics
```python
from mayini.training import Metrics

# Classification metrics
accuracy = Metrics.accuracy(predictions, targets)
precision, recall, f1 = Metrics.precision_recall_f1(predictions, targets, num_classes=10)
cm = Metrics.confusion_matrix(predictions, targets, num_classes=10)

# Regression metrics
mse = Metrics.mse(predictions, targets)
mae = Metrics.mae(predictions, targets)
r2 = Metrics.r2_score(predictions, targets)
```
#### evaluate()
```python
results = trainer.evaluate(
    test_loader,    # Test data loader
    detailed=True   # Compute detailed metrics
)
```
#### predict()
```python
predictions = trainer.predict(X)  # Returns numpy array
```

#### Early Stopping
```python
from mayini.training import EarlyStopping

early_stopping = EarlyStopping(
    patience=7,
    min_delta=0.0,
    restore_best_weights=True,
    mode='min'  # 'min' for loss, 'max' for accuracy
)

history = trainer.fit(
    train_loader,
    epochs=100,
    val_loader=val_loader,
    early_stopping=early_stopping
)
```
---
#### Metrics
```python
from mayini.training import Metrics
```
#### accuracy()
```python
accuracy = Metrics.accuracy(predictions, targets)
# Returns: float (0.0 to 1.0)
```
#### precision_recall_f1()
```python
precision, recall, f1 = Metrics.precision_recall_f1(
    predictions, 
    targets, 
    num_classes=10
)
# Returns: Three numpy arrays of shape (num_classes,)
```
#### confusion_matrix()
```python
cm = Metrics.confusion_matrix(predictions, targets, num_classes=10)
# Returns: numpy array of shape (num_classes, num_classes)
```
#### r2_score()
```python
r2 = Metrics.r2_score(predictions, targets)
```
---

## 💡 Complete Examples

### Example 1: Basic Training

```python
import numpy as np
import mayini as mn
from mayini.nn import Sequential, Linear, ReLU, Softmax, CrossEntropyLoss
from mayini.optim import Adam
from mayini.training import DataLoader, Trainer

# Build model
model = Sequential(
    Linear(784, 128, init_method='he'),
    ReLU(),
    Linear(128, 10),
    Softmax(dim=1)
)

# Prepare data
X_train = np.random.randn(5000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 5000)

train_loader = DataLoader(X_train, y_train, batch_size=128, shuffle=True)

# Train
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()
trainer = Trainer(model, optimizer, criterion)

history = trainer.fit(train_loader, epochs=20, verbose=True)
```

### Example 2: MNIST Classification

```python
import mayini as mn
import numpy as np
from mayini.nn import Sequential, Linear, ReLU, Dropout, Softmax, CrossEntropyLoss
from mayini.optim import Adam
from mayini.training import DataLoader, Trainer

# Build model
model = Sequential(
    Linear(784, 512, init_method='he'),
    ReLU(),
    Dropout(0.2),
    Linear(512, 256, init_method='he'),
    ReLU(),
    Dropout(0.2),
    Linear(256, 10),
    Softmax(dim=1)
)

# Prepare data
X_train = np.random.randn(5000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 5000)
X_val = np.random.randn(1000, 784).astype(np.float32)
y_val = np.random.randint(0, 10, 1000)

train_loader = DataLoader(X_train, y_train, batch_size=128, shuffle=True)
val_loader = DataLoader(X_val, y_val, batch_size=128, shuffle=False)

# Train
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()
trainer = Trainer(model, optimizer, criterion)

history = trainer.fit(train_loader, epochs=20, val_loader=val_loader, verbose=True)
```

### Example 3: CNN for Image Classification

```python
from mayini.nn import Conv2D, MaxPool2D, Flatten, BatchNorm1d

cnn_model = Sequential(
    # Conv block 1
    Conv2D(1, 32, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(kernel_size=2, stride=2),
    
    # Conv block 2
    Conv2D(32, 64, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(kernel_size=2, stride=2),
    
    # Classifier
    Flatten(),
    Linear(64 * 7 * 7, 256),
    ReLU(),
    Dropout(0.5),
    Linear(256, 10),
    Softmax(dim=1)
)

# Train similarly to Example 1
```

### Example 4: LSTM for Sequence Classification

```python
from mayini.nn import RNN

lstm_model = Sequential(
    RNN(
        input_size=100,
        hidden_size=128,
        num_layers=2,
        cell_type='lstm',
        dropout=0.3,
        batch_first=True
    ),
    Linear(128, 64),
    ReLU(),
    Linear(64, 3),
    Softmax(dim=1)
)

# Process sequences (batch, seq_len, features)
x_seq = mn.Tensor(np.random.randn(32, 50, 100))
output, _ = lstm_model(x_seq)
```

### Example 5: Custom Training Loop

```python
# Manual training loop with learning rate scheduling
from mayini.optim import Adam, StepLR

optimizer = Adam(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = CrossEntropyLoss()

for epoch in range(50):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Update learning rate
    scheduler.step()
    
    print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}, LR = {optimizer.lr:.6f}")
```
#### Example 6:Training with validation
```python
import numpy as np
from mayini.nn import Sequential, Linear, ReLU, Dropout, Softmax, CrossEntropyLoss
from mayini.optim import Adam
from mayini.training import DataLoader, Trainer

# Build model with dropout
model = Sequential(
    Linear(784, 512, init_method='he'),
    ReLU(),
    Dropout(0.3),
    Linear(512, 256, init_method='he'),
    ReLU(),
    Dropout(0.3),
    Linear(256, 10),
    Softmax(dim=1)
)

# Prepare train and validation data
X_train = np.random.randn(5000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 5000)
X_val = np.random.randn(1000, 784).astype(np.float32)
y_val = np.random.randint(0, 10, 1000)

train_loader = DataLoader(X_train, y_train, batch_size=128, shuffle=True)
val_loader = DataLoader(X_val, y_val, batch_size=128, shuffle=False)

# Train with validation
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()
trainer = Trainer(model, optimizer, criterion)

history = trainer.fit(
    train_loader,
    epochs=30,
    val_loader=val_loader,
    verbose=True
)

# Plot training curves (if matplotlib available)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.show()
```
### Example 7: Evaluation and Testing
```python
import numpy as np
from mayini.training import Trainer, DataLoader, Metrics

# Assume model is already trained (from previous examples)

# Prepare test data
X_test = np.random.randn(1000, 784).astype(np.float32)
y_test = np.random.randint(0, 10, 1000)
test_loader = DataLoader(X_test, y_test, batch_size=128, shuffle=False)

# Evaluate
results = trainer.evaluate(test_loader, detailed=True)

print("Test Results:")
print(f"Test Loss: {results['test_loss']:.4f}")
print(f"Test Accuracy: {results['accuracy']:.4f}")

print("\nPer-class Metrics:")
for i in range(10):
    print(f"Class {i}:")
    print(f"  Precision: {results['precision'][i]:.3f}")
    print(f"  Recall:    {results['recall'][i]:.3f}")
    print(f"  F1-Score:  {results['f1_score'][i]:.3f}")

print("\nConfusion Matrix:")
print(results['confusion_matrix'])

# Make predictions on new data
X_new = np.random.randn(10, 784).astype(np.float32)
predictions = trainer.predict(X_new)
predicted_classes = np.argmax(predictions, axis=1)
print(f"\nPredicted classes: {predicted_classes}")
```
### Example 8: Custom Training Loop
```python
import numpy as np
from mayini.nn import Sequential, Linear, ReLU, Softmax, CrossEntropyLoss
from mayini.optim import Adam
from mayini.training import DataLoader
import mayini as mn

# Build model
model = Sequential(
    Linear(784, 256, init_method='he'),
    ReLU(),
    Linear(256, 10),
    Softmax(dim=1)
)

# Prepare data
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)
train_loader = DataLoader(X_train, y_train, batch_size=64, shuffle=True)

# Setup
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# Custom training loop
history = {'train_loss': [], 'train_acc': []}

for epoch in range(20):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        epoch_loss += loss.item()
        pred_classes = np.argmax(predictions.data, axis=1)
        correct += np.sum(pred_classes == batch_y.data.flatten())
        total += len(batch_y.data)
    
    # Calculate epoch metrics
    avg_loss = epoch_loss / len(train_loader)
    accuracy = correct / total
    
    history['train_loss'].append(avg_loss)
    history['train_acc'].append(accuracy)
    
    print(f"Epoch {epoch+1}/20 - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
```
---

## 📂 Module Structure

```
mayini/
├── __init__.py           # Main package
├── tensor.py             # Tensor with autograd
├── nn/
│   ├── modules.py        # Layers (Linear, Conv2D, etc.)
│   ├── activations.py    # Activation functions
│   ├── losses.py         # Loss functions
│   └── rnn.py            # RNN components
├── optim/
│   └── optimizers.py     # Optimizers & LR schedulers
└── training/
    └── trainer.py        # Training utilities
```

---

## 🎓 Educational Resources

### Interactive Notebook
**[Open in Google Colab](https://colab.research.google.com/drive/140HDqQ3vBGy6HIpzbvH8Jv54PeLylNOK?usp=sharing)**

The notebook includes 38 runnable examples covering:
- Tensor operations and autograd
- All neural network layers
- All activation functions
- RNN/LSTM/GRU cells
- Loss functions
- Optimizers and schedulers
- Complete training workflows
- CNN and LSTM projects

### Key Concepts

**Automatic Differentiation:**  
MAYINI implements reverse-mode automatic differentiation (backpropagation) with computational graph construction and cycle detection.

**Initialization Methods:**
- Xavier/Glorot: Good for sigmoid/tanh activations
- He: Recommended for ReLU activations
- Normal: Simple normal distribution

**Training Best Practices:**
1. Use He initialization with ReLU
2. Apply batch normalization for deep networks
3. Use dropout for regularization
4. Start with Adam optimizer
5. Apply learning rate scheduling
6. Monitor validation metrics
7. Use early stopping to prevent overfitting

---

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=mayini tests/
```

---

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by PyTorch's design philosophy
- Built for educational purposes and research
- Thanks to the open-source community

---

## 📞 Support & Links

- **GitHub Repository:** [907-bot-collab/mayini](https://github.com/907-bot-collab/mayini)
- **PyPI Package:** [mayini-framework](https://pypi.org/project/mayini-framework)
- **Interactive Notebook:** [Google Colab](https://colab.research.google.com/drive/140HDqQ3vBGy6HIpzbvH8Jv54PeLylNOK?usp=sharing)
- **Report Issues:** [GitHub Issues](https://github.com/907-bot-collab/mayini/issues)
- **Documentation:** This README

---

## 🗺️ Version History

- **v0.1.9** (Latest): Fixed Module.__call__(), exported LR schedulers, removed numpy upper bound
- **v0.1.8**: Added comprehensive RNN support
- **v0.1.7**: Initial public release
- **v0.1.6**: Beta release

---

## 🎯 Comparison with Other Frameworks

| Feature | MAYINI | PyTorch | TensorFlow |
|---------|--------|---------|------------|
| Educational Focus | ✅ | ❌ | ❌ |
| Transparent Implementation | ✅ | ❌ | ❌ |
| Automatic Differentiation | ✅ | ✅ | ✅ |
| GPU Support | ❌ | ✅ | ✅ |
| Production Ready | ❌ | ✅ | ✅ |
| Easy to Understand | ✅ | ⚠️ | ❌ |
| From-Scratch Implementation | ✅ | ❌ | ❌ |

---

## 💻 Quick Reference

### Essential Imports
```python
import mayini as mn
from mayini.nn import (
    Sequential, Linear, Conv2D, MaxPool2D, Flatten,
    ReLU, Sigmoid, Tanh, Softmax,
    RNN, LSTMCell, GRUCell,
    MSELoss, CrossEntropyLoss
)
from mayini.optim import Adam, SGD, StepLR
from mayini.training import DataLoader, Trainer, Metrics, EarlyStopping
```

### Minimal Working Example
```python
import mayini as mn
import numpy as np
from mayini.nn import Sequential, Linear, ReLU, Softmax, CrossEntropyLoss
from mayini.optim import Adam
from mayini.training import DataLoader, Trainer

# Model
model = Sequential(Linear(10, 5), ReLU(), Linear(5, 2), Softmax(dim=1))

# Data
X = np.random.randn(100, 10).astype(np.float32)
y = np.random.randint(0, 2, 100)
loader = DataLoader(X, y, batch_size=32)

# Train
trainer = Trainer(model, Adam(model.parameters(), lr=0.01), CrossEntropyLoss())
history = trainer.fit(loader, epochs=10)
```

---

**MAYINI** - Making AI Neural Intelligence Intuitive 🧠✨

Built with ❤️ for education and research | [Try it now in Colab!](https://colab.research.google.com/drive/140HDqQ3vBGy6HIpzbvH8Jv54PeLylNOK?usp=sharing)
