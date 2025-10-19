# MAYINI Deep Learning Framework

[![PyPI version](https://badge.fury.io/py/mayini-framework.svg)](https://badge.fury.io/py/mayini-framework)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/907-bot-collab/mayini/workflows/CI/badge.svg)](https://github.com/907-bot-collab/mayini/actions)

MAYINI is a comprehensive deep learning framework built from scratch in Python, featuring automatic differentiation, neural network components, and complete training infrastructure. It's designed for educational purposes and research, providing a PyTorch-like API with full transparency into the underlying mechanics.

## 🚀 Key Features

### Core Engine
- **Tensor Operations**: Complete tensor class with automatic differentiation
- **Computational Graph**: Cycle detection and gradient computation
- **Broadcasting Support**: NumPy-style broadcasting for operations

### Neural Network Components
- **Linear Layers**: Dense layers with multiple initialization methods (Xavier, He, Normal)
- **Convolutional Layers**: 2D convolution with im2col optimization
- **Pooling Layers**: Max and Average pooling with stride and padding support
- **Normalization**: Batch Normalization for improved training
- **Regularization**: Dropout with inverted dropout implementation

### Activation Functions
- **Standard Functions**: ReLU, Sigmoid, Tanh, Softmax
- **Modern Activations**: GELU, Leaky ReLU
- **Numerical Stability**: Implemented with overflow/underflow protection

### Recurrent Neural Networks
- **RNN Cells**: Vanilla RNN with configurable activations
- **LSTM Cells**: Long Short-Term Memory with proper gate mechanisms
- **GRU Cells**: Gated Recurrent Units for efficient sequence modeling
- **Multi-layer Support**: Stack multiple RNN layers with dropout

### Loss Functions
- **Regression**: MSE Loss, MAE Loss, Huber Loss
- **Classification**: Cross-Entropy Loss, Binary Cross-Entropy Loss
- **Flexible Reduction**: Support for mean, sum, and none reduction modes

### Optimization Algorithms
- **SGD**: Stochastic Gradient Descent with momentum and weight decay
- **Adam**: Adaptive moment estimation with bias correction
- **AdamW**: Adam with decoupled weight decay
- **RMSprop**: Root Mean Square Propagation
- **Learning Rate Schedulers**: Step, Exponential, and Cosine Annealing schedulers

### Training Infrastructure
- **DataLoader**: Efficient batch processing with shuffling
- **Trainer**: Complete training framework with logging and checkpointing
- **Metrics**: Comprehensive evaluation (accuracy, precision, recall, F1)
- **Early Stopping**: Prevent overfitting with validation monitoring
- **Checkpointing**: Save and restore model states

## 📦 Installation

### From PyPI
```bash
pip install mayini-framework
```

### From Source
```bash
git clone https://github.com/907-bot-collab/mayini.git
cd mayini
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/907-bot-collab/mayini.git
cd mayini
pip install -e ".[dev]"
```

---

## 📚 Complete Documentation

## 1. Core Components

### 1.1 Tensor Operations

The `Tensor` class is the fundamental building block with automatic differentiation support.

```python
import mayini as mn
import numpy as np

# Create tensors with gradient tracking
x = mn.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = mn.Tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)

# Tensor operations
z = x.matmul(y)      # Matrix multiplication
w = x + y            # Element-wise addition
v = x * 2            # Scalar multiplication
u = x ** 2           # Power operation

# Reduction operations
sum_val = x.sum()             # Sum all elements
mean_val = x.mean()           # Mean of all elements
sum_axis = x.sum(axis=0)      # Sum along axis

# Shape operations
reshaped = x.reshape((4, 1))  # Reshape tensor
transposed = x.transpose()    # Transpose tensor

# Automatic differentiation
loss = z.sum()
loss.backward()
print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")
```

**Key Methods:**
- `matmul(other)`: Matrix multiplication
- `sum(axis=None, keepdims=False)`: Sum reduction
- `mean(axis=None, keepdims=False)`: Mean reduction
- `reshape(shape)`: Reshape tensor
- `transpose(axes=None)`: Transpose dimensions
- `backward(gradient=None)`: Compute gradients
- `zero_grad()`: Reset gradients
- `detach()`: Detach from computation graph
- `numpy()`: Convert to numpy array
- `item()`: Get scalar value

---

## 2. Neural Network Layers

### 2.1 Linear (Fully Connected) Layer

Dense layer with multiple weight initialization methods.

```python
from mayini.nn import Linear

# Create linear layers with different initializations
layer1 = Linear(784, 256, init_method='xavier')  # Xavier initialization
layer2 = Linear(256, 128, init_method='he')      # He initialization  
layer3 = Linear(128, 10, init_method='normal')   # Normal initialization

# Forward pass
x = mn.Tensor(np.random.randn(32, 784))
output = layer1(x)
print(f"Output shape: {output.shape}")  # (32, 256)
```

**Parameters:**
- `in_features`: Number of input features
- `out_features`: Number of output features
- `bias`: Whether to include bias term (default: True)
- `init_method`: Weight initialization ('xavier', 'he', 'normal')

### 2.2 Convolutional Layer (Conv2D)

2D convolutional layer with im2col optimization for efficiency.

```python
from mayini.nn import Conv2D

# Create convolutional layer
conv = Conv2D(
    in_channels=3,      # RGB input
    out_channels=64,    # 64 filters
    kernel_size=3,      # 3x3 kernel
    stride=1,           # Stride of 1
    padding=1,          # Padding of 1
    bias=True           # Include bias
)

# Forward pass
x = mn.Tensor(np.random.randn(32, 3, 28, 28))  # (batch, channels, height, width)
output = conv(x)
print(f"Output shape: {output.shape}")  # (32, 64, 28, 28)
```

**Parameters:**
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels (filters)
- `kernel_size`: Size of convolutional kernel
- `stride`: Stride of convolution (default: 1)
- `padding`: Zero padding (default: 0)
- `bias`: Whether to include bias (default: True)

### 2.3 Pooling Layers

#### MaxPool2D
```python
from mayini.nn import MaxPool2D

pool = MaxPool2D(kernel_size=2, stride=2, padding=0)
x = mn.Tensor(np.random.randn(32, 64, 28, 28))
output = pool(x)
print(f"Output shape: {output.shape}")  # (32, 64, 14, 14)
```

#### AvgPool2D
```python
from mayini.nn import AvgPool2D

pool = AvgPool2D(kernel_size=2, stride=2, padding=0)
x = mn.Tensor(np.random.randn(32, 64, 28, 28))
output = pool(x)
print(f"Output shape: {output.shape}")  # (32, 64, 14, 14)
```

### 2.4 Normalization Layers

#### Batch Normalization
```python
from mayini.nn import BatchNorm1d

bn = BatchNorm1d(num_features=256, eps=1e-5, momentum=0.1)

# Training mode
bn.train()
x = mn.Tensor(np.random.randn(32, 256))
output = bn(x)

# Evaluation mode
bn.eval()
x_test = mn.Tensor(np.random.randn(16, 256))
output_test = bn(x_test)
```

**Parameters:**
- `num_features`: Number of features/channels
- `eps`: Small constant for numerical stability (default: 1e-5)
- `momentum`: Momentum for running statistics (default: 0.1)

### 2.5 Regularization

#### Dropout
```python
from mayini.nn import Dropout

dropout = Dropout(p=0.5)

# Training mode - drops 50% of neurons
dropout.train()
x = mn.Tensor(np.random.randn(32, 256))
output = dropout(x)

# Evaluation mode - no dropout
dropout.eval()
output_test = dropout(x)
```

### 2.6 Utility Layers

#### Flatten
```python
from mayini.nn import Flatten

flatten = Flatten(start_dim=1)
x = mn.Tensor(np.random.randn(32, 64, 7, 7))
output = flatten(x)
print(f"Output shape: {output.shape}")  # (32, 3136)
```

---

## 3. Activation Functions

### 3.1 ReLU (Rectified Linear Unit)

```python
from mayini.nn import ReLU

relu = ReLU()
x = mn.Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
output = relu(x)
print(output.data)  # [[0. 0. 0. 1. 2.]]
```

**Formula:** \( f(x) = \max(0, x) \)

**Use cases:** Most common activation for hidden layers in deep networks

### 3.2 Sigmoid

```python
from mayini.nn import Sigmoid

sigmoid = Sigmoid()
x = mn.Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
output = sigmoid(x)
print(output.data)  # [[0.119 0.269 0.5 0.731 0.881]]
```

**Formula:** \( f(x) = \frac{1}{1 + e^{-x}} \)

**Use cases:** Binary classification output layer, gate activations in LSTM

### 3.3 Tanh (Hyperbolic Tangent)

```python
from mayini.nn import Tanh

tanh = Tanh()
x = mn.Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
output = tanh(x)
print(output.data)  # [[-0.964 -0.762 0. 0.762 0.964]]
```

**Formula:** \( f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)

**Use cases:** Hidden layers in RNNs, when zero-centered activations are needed

### 3.4 Softmax

```python
from mayini.nn import Softmax

softmax = Softmax(dim=1)
x = mn.Tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
output = softmax(x)
print(output.data)  
# [[0.090 0.245 0.665]
#  [0.333 0.333 0.333]]
```

**Formula:** \( f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} \)

**Use cases:** Multi-class classification output layer

### 3.5 GELU (Gaussian Error Linear Unit)

```python
from mayini.nn import GELU

gelu = GELU()
x = mn.Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
output = gelu(x)
```

**Formula:** \( f(x) = 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3))) \)

**Use cases:** Modern transformer models, BERT, GPT

### 3.6 Leaky ReLU

```python
from mayini.nn import LeakyReLU

leaky_relu = LeakyReLU(negative_slope=0.01)
x = mn.Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
output = leaky_relu(x)
print(output.data)  # [[-0.02 -0.01 0. 1. 2.]]
```

**Formula:** \( f(x) = \max(\alpha x, x) \) where \( \alpha = 0.01 \)

**Use cases:** Alternative to ReLU to prevent dead neurons

---

## 4. Recurrent Neural Networks

### 4.1 RNN Cell (Vanilla RNN)

```python
from mayini.nn import RNNCell

rnn_cell = RNNCell(input_size=100, hidden_size=128, bias=True)

# Single timestep
x_t = mn.Tensor(np.random.randn(32, 100))  # (batch_size, input_size)
h_t = mn.Tensor(np.random.randn(32, 128))  # (batch_size, hidden_size)

h_next = rnn_cell(x_t, h_t)
print(f"Next hidden state: {h_next.shape}")  # (32, 128)
```

**Formula:** \( h_t = \tanh(W_{ih}x_t + b_{ih} + W_{hh}h_{t-1} + b_{hh}) \)

**Parameters:**
- `input_size`: Size of input features
- `hidden_size`: Size of hidden state
- `bias`: Whether to include bias (default: True)

### 4.2 LSTM Cell (Long Short-Term Memory)

```python
from mayini.nn import LSTMCell

lstm_cell = LSTMCell(input_size=100, hidden_size=128, bias=True)

# Single timestep
x_t = mn.Tensor(np.random.randn(32, 100))
h_t = mn.Tensor(np.random.randn(32, 128))
c_t = mn.Tensor(np.random.randn(32, 128))

h_next, c_next = lstm_cell(x_t, (h_t, c_t))
print(f"Next hidden: {h_next.shape}, Next cell: {c_next.shape}")
```

**Gates:**
- **Forget gate:** \( f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \)
- **Input gate:** \( i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \)
- **Output gate:** \( o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \)
- **Cell candidate:** \( \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \)
- **Cell state:** \( C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \)
- **Hidden state:** \( h_t = o_t \odot \tanh(C_t) \)

**Use cases:** Long sequence modeling, machine translation, speech recognition

### 4.3 GRU Cell (Gated Recurrent Unit)

```python
from mayini.nn import GRUCell

gru_cell = GRUCell(input_size=100, hidden_size=128, bias=True)

# Single timestep
x_t = mn.Tensor(np.random.randn(32, 100))
h_t = mn.Tensor(np.random.randn(32, 128))

h_next = gru_cell(x_t, h_t)
print(f"Next hidden state: {h_next.shape}")  # (32, 128)
```

**Gates:**
- **Reset gate:** \( r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) \)
- **Update gate:** \( z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) \)
- **New gate:** \( \tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t]) \)
- **Hidden state:** \( h_t = (1 - z_t) \odot \tilde{h}_t + z_t \odot h_{t-1} \)

**Use cases:** More efficient than LSTM, good for shorter sequences

### 4.4 Multi-layer RNN

```python
from mayini.nn import RNN

# Multi-layer LSTM
lstm_model = RNN(
    input_size=100,
    hidden_size=128,
    num_layers=2,
    cell_type='lstm',      # 'rnn', 'lstm', or 'gru'
    dropout=0.2,
    batch_first=True
)

# Process sequences
x_seq = mn.Tensor(np.random.randn(32, 50, 100))  # (batch, seq_len, features)
output, hidden_states = lstm_model(x_seq)

print(f"Output shape: {output.shape}")  # (32, 50, 128)
print(f"Number of hidden states: {len(hidden_states)}")  # 2 (one per layer)
```

**Parameters:**
- `input_size`: Size of input features
- `hidden_size`: Size of hidden state
- `num_layers`: Number of stacked RNN layers
- `cell_type`: Type of RNN cell ('rnn', 'lstm', or 'gru')
- `dropout`: Dropout between RNN layers (default: 0.0)
- `batch_first`: If True, input shape is (batch, seq, features)

---

## 5. Loss Functions

### 5.1 MSE Loss (Mean Squared Error)

```python
from mayini.nn import MSELoss

criterion = MSELoss(reduction='mean')  # 'mean', 'sum', or 'none'

predictions = mn.Tensor([[1.0, 2.0], [3.0, 4.0]])
targets = mn.Tensor([[1.5, 2.5], [3.5, 4.5]])

loss = criterion(predictions, targets)
print(f"MSE Loss: {loss.item()}")  # 0.25
```

**Formula:** \( L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 \)

**Use cases:** Regression tasks, predicting continuous values

### 5.2 MAE Loss (Mean Absolute Error)

```python
from mayini.nn import MAELoss

criterion = MAELoss(reduction='mean')

predictions = mn.Tensor([[1.0, 2.0], [3.0, 4.0]])
targets = mn.Tensor([[1.5, 2.5], [3.5, 4.5]])

loss = criterion(predictions, targets)
print(f"MAE Loss: {loss.item()}")  # 0.5
```

**Formula:** \( L = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| \)

**Use cases:** Regression with outliers, robust to outliers than MSE

### 5.3 Cross-Entropy Loss

```python
from mayini.nn import CrossEntropyLoss

criterion = CrossEntropyLoss(reduction='mean')

# Predictions (before softmax): (batch_size, num_classes)
predictions = mn.Tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]])
# Targets (class indices): (batch_size,)
targets = mn.Tensor([0, 1])  # Class 0 and Class 1

loss = criterion(predictions, targets)
print(f"Cross-Entropy Loss: {loss.item()}")
```

**Formula:** \( L = -\frac{1}{n}\sum_{i=1}^{n}\log(\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}) \)

**Use cases:** Multi-class classification

### 5.4 Binary Cross-Entropy Loss

```python
from mayini.nn import BCELoss

criterion = BCELoss(reduction='mean')

# Predictions (after sigmoid): (batch_size,)
predictions = mn.Tensor([0.8, 0.3, 0.6])
# Targets (0 or 1): (batch_size,)
targets = mn.Tensor([1.0, 0.0, 1.0])

loss = criterion(predictions, targets)
print(f"BCE Loss: {loss.item()}")
```

**Formula:** \( L = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)] \)

**Use cases:** Binary classification

### 5.5 Huber Loss

```python
from mayini.nn import HuberLoss

criterion = HuberLoss(delta=1.0, reduction='mean')

predictions = mn.Tensor([[1.0, 2.0], [3.0, 4.0]])
targets = mn.Tensor([[1.5, 2.5], [5.0, 6.0]])

loss = criterion(predictions, targets)
print(f"Huber Loss: {loss.item()}")
```

**Formula:**
\[
L_\delta(y, \hat{y}) = \begin{cases} 
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
\]

**Use cases:** Robust regression, less sensitive to outliers than MSE

---

## 6. Optimizers

### 6.1 SGD (Stochastic Gradient Descent)

```python
from mayini.optim import SGD

optimizer = SGD(
    model.parameters(), 
    lr=0.01,              # Learning rate
    momentum=0.9,         # Momentum factor
    weight_decay=1e-4     # L2 regularization
)

# Training loop
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Update rule:**
\[
v_t = \beta v_{t-1} + g_t \\
\theta_t = \theta_{t-1} - \eta v_t
\]

**Parameters:**
- `lr`: Learning rate
- `momentum`: Momentum factor (default: 0.0)
- `weight_decay`: L2 penalty (default: 0.0)

### 6.2 Adam (Adaptive Moment Estimation)

```python
from mayini.optim import Adam

optimizer = Adam(
    model.parameters(),
    lr=0.001,           # Learning rate
    beta1=0.9,          # Exponential decay rate for 1st moment
    beta2=0.999,        # Exponential decay rate for 2nd moment
    eps=1e-8,           # Small constant for numerical stability
    weight_decay=0.0    # L2 regularization
)
```

**Update rule:**
\[
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \\
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \\
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]

**Use cases:** Default choice for most deep learning tasks

### 6.3 AdamW (Adam with Decoupled Weight Decay)

```python
from mayini.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.01   # Decoupled weight decay
)
```

**Use cases:** Better generalization than Adam, recommended for transformers

### 6.4 RMSprop (Root Mean Square Propagation)

```python
from mayini.optim import RMSprop

optimizer = RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,         # Smoothing constant
    eps=1e-8,
    momentum=0.0,
    weight_decay=0.0
)
```

**Update rule:**
\[
v_t = \alpha v_{t-1} + (1-\alpha)g_t^2 \\
\theta_t = \theta_{t-1} - \eta \frac{g_t}{\sqrt{v_t} + \epsilon}
\]

**Use cases:** Good for RNNs, online learning

---

## 7. Learning Rate Schedulers

### 7.1 StepLR

```python
from mayini.optim import Adam, StepLR

optimizer = Adam(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(50):
    train(...)
    scheduler.step()  # Decay LR every 10 epochs
```

**Schedule:** LR = base_lr \* gamma^(epoch // step_size)

### 7.2 ExponentialLR

```python
from mayini.optim import ExponentialLR

optimizer = Adam(model.parameters(), lr=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.95)

for epoch in range(50):
    train(...)
    scheduler.step()  # LR = base_lr * 0.95^epoch
```

### 7.3 CosineAnnealingLR

```python
from mayini.optim import CosineAnnealingLR

optimizer = Adam(model.parameters(), lr=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

for epoch in range(50):
    train(...)
    scheduler.step()  # Cosine decay
```

**Schedule:** \( \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t\pi}{T_{max}})) \)

---

## 8. Training Utilities

### 8.1 DataLoader

```python
from mayini.training import DataLoader
import numpy as np

# Create dataset
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)

# Create data loader
train_loader = DataLoader(
    X_train, 
    y_train, 
    batch_size=64, 
    shuffle=True
)

# Iterate through batches
for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
    print(f"Batch {batch_idx}: X shape {batch_X.shape}, y shape {batch_y.shape}")
```

**Parameters:**
- `X`: Input features (numpy array or Tensor)
- `y`: Target labels (numpy array or Tensor)
- `batch_size`: Number of samples per batch (default: 32)
- `shuffle`: Whether to shuffle data each epoch (default: True)

### 8.2 Trainer

Complete training framework with automatic logging, checkpointing, and validation.

```python
from mayini.training import Trainer
from mayini.optim import Adam
from mayini.nn import CrossEntropyLoss

# Setup
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

trainer = Trainer(model, optimizer, criterion)

# Train the model
history = trainer.fit(
    train_loader,
    epochs=10,
    val_loader=val_loader,      # Optional validation data
    early_stopping=None,         # Optional early stopping
    verbose=True,                # Print progress
    save_best=True,              # Save best model
    checkpoint_path='model.pkl'  # Checkpoint file path
)

# Access training history
print(f"Training losses: {history['train_loss']}")
print(f"Validation losses: {history['val_loss']}")
print(f"Training accuracy: {history['train_acc']}")
print(f"Validation accuracy: {history['val_acc']}")
```

**Methods:**
- `fit()`: Train the model
- `evaluate()`: Evaluate on test data
- `predict()`: Make predictions
- `save_checkpoint()`: Save model state
- `load_checkpoint()`: Load model state

### 8.3 Metrics

```python
from mayini.training import Metrics

# Accuracy
accuracy = Metrics.accuracy(predictions, targets)

# Precision, Recall, F1-score
precision, recall, f1 = Metrics.precision_recall_f1(
    predictions, targets, num_classes=10
)

# Confusion matrix
cm = Metrics.confusion_matrix(predictions, targets, num_classes=10)

# Regression metrics
mse = Metrics.mse(predictions, targets)
mae = Metrics.mae(predictions, targets)
r2 = Metrics.r2_score(predictions, targets)
```

**Available Metrics:**
- `accuracy()`: Classification accuracy
- `precision_recall_f1()`: Per-class precision, recall, F1
- `confusion_matrix()`: Confusion matrix
- `mse()`: Mean squared error
- `mae()`: Mean absolute error
- `r2_score()`: R² score for regression

### 8.4 Early Stopping

```python
from mayini.training import EarlyStopping

early_stopping = EarlyStopping(
    patience=7,                  # Number of epochs to wait
    min_delta=0.001,             # Minimum improvement threshold
    restore_best_weights=True,   # Restore best weights when stopped
    mode='min'                   # 'min' for loss, 'max' for accuracy
)

# Use with trainer
history = trainer.fit(
    train_loader,
    epochs=100,
    val_loader=val_loader,
    early_stopping=early_stopping,
    verbose=True
)
```

---

## 9. Complete Examples

### 9.1 Simple Neural Network for MNIST

```python
import mayini as mn
import numpy as np
from mayini.nn import Sequential, Linear, ReLU, Softmax, CrossEntropyLoss, Dropout
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

# Setup training
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()
trainer = Trainer(model, optimizer, criterion)

# Train
history = trainer.fit(
    train_loader, 
    epochs=20,
    val_loader=val_loader,
    verbose=True
)
```

### 9.2 CNN for Image Classification

```python
from mayini.nn import Conv2D, MaxPool2D, Flatten, BatchNorm1d

# Build CNN
cnn_model = Sequential(
    # Conv block 1
    Conv2D(1, 32, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(kernel_size=2, stride=2),
    
    # Conv block 2
    Conv2D(32, 64, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(kernel_size=2, stride=2),
    
    # Flatten and FC layers
    Flatten(),
    Linear(64 * 7 * 7, 256),
    ReLU(),
    Dropout(0.5),
    Linear(256, 10),
    Softmax(dim=1)
)

# Train (same as above)
optimizer = Adam(cnn_model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()
trainer = Trainer(cnn_model, optimizer, criterion)
```

### 9.3 LSTM for Sequence Classification

```python
from mayini.nn import RNN

# Build LSTM model
lstm_model = Sequential(
    RNN(
        input_size=100,
        hidden_size=128,
        num_layers=2,
        cell_type='lstm',
        dropout=0.3,
        batch_first=True
    ),
    # Extract last timestep output (implement custom layer or use slicing)
    Linear(128, 64),
    ReLU(),
    Linear(64, 3),  # 3 classes
    Softmax(dim=1)
)
```

### 9.4 GRU for Time Series Prediction

```python
# Build GRU model for regression
gru_model = Sequential(
    RNN(
        input_size=10,
        hidden_size=64,
        num_layers=3,
        cell_type='gru',
        dropout=0.2,
        batch_first=True
    ),
    Linear(64, 32),
    ReLU(),
    Linear(32, 1)  # Single output for regression
)

# Use MSE loss for regression
criterion = MSELoss()
optimizer = Adam(gru_model.parameters(), lr=0.001)
```

---

## 10. Module Structure

```
mayini/
├── __init__.py           # Main package initialization
├── tensor.py             # Core Tensor class with autograd
├── base.py               # Base classes
├── metrics.py            # Standalone metrics utilities
├── utils.py              # Utility functions
├── nn/
│   ├── __init__.py
│   ├── modules.py        # Module, Sequential, Linear, Conv2D, etc.
│   ├── activations.py    # ReLU, Sigmoid, Tanh, Softmax, GELU, etc.
│   ├── losses.py         # MSELoss, CrossEntropyLoss, BCELoss, etc.
│   └── rnn.py            # RNNCell, LSTMCell, GRUCell, RNN
├── optim/
│   ├── __init__.py
│   └── optimizers.py     # SGD, Adam, AdamW, RMSprop, Schedulers
└── training/
    ├── __init__.py
    └── trainer.py        # DataLoader, Trainer, Metrics, EarlyStopping
```

---

## 11. Quick Reference

### Imports
```python
import mayini as mn
from mayini.nn import (
    Sequential, Module,
    Linear, Conv2D, MaxPool2D, AvgPool2D, Flatten,
    BatchNorm1d, Dropout,
    ReLU, Sigmoid, Tanh, Softmax, GELU, LeakyReLU,
    RNN, RNNCell, LSTMCell, GRUCell,
    MSELoss, MAELoss, CrossEntropyLoss, BCELoss, HuberLoss
)
from mayini.optim import SGD, Adam, AdamW, RMSprop
from mayini.optim import StepLR, ExponentialLR, CosineAnnealingLR
from mayini.training import DataLoader, Trainer, Metrics, EarlyStopping
```

---

## 🧪 Testing

```bash
pytest tests/
pytest --cov=mayini tests/
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Inspired by PyTorch's design philosophy
- Built for educational purposes and research

## 📞 Support

- **GitHub**: [907-bot-collab/mayini](https://github.com/907-bot-collab/mayini)
- **PyPI**: [mayini-framework](https://pypi.org/project/mayini-framework)
- **Issues**: [Report Issues](https://github.com/907-bot-collab/mayini/issues)
- **Colab**: [Example Notebook](https://colab.research.google.com/drive/140HDqQ3vBGy6HIpzbvH8Jv54PeLylNOK)

---

**MAYINI** - Making AI Neural Intelligence Intuitive 🧠✨
