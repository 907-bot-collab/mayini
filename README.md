# Mayini Framework
![PyPI](https://img.shields.io/pypi/v/mayini-framework?color=blue&label=pypi%20version)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Test Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

**Mayini** is a comprehensive, from-scratch Deep Learning and Machine Learning framework built in pure Python. Designed with a clean, PyTorch-like API, it aims to demystify underlying AI mechanics while providing robust, high-performance features for deep neural networks, classical ML models, evolutionary algorithms, and multimodal data preprocessing.

Whether you're researching new topologies, learning the mathematics of backpropagation, or deploying classical models, Mayini has you covered.

---

## 📦 Installation

Install Mayini directly from PyPI via pip:

```bash
pip install mayini-framework==0.8.1
```

If you wish to edit the source code and contribute:
```bash
git clone https://github.com/your-username/mayini.git
cd mayini
pip install -e .[dev]
```

---

## 🚀 Key Modules & Architecture

The framework is divided into five highly independent yet composable pillars:

### 1. 🧠 `mayini.tensor` (The Autograd Engine)
A custom multidimensional array computing engine built on top of NumPy. It features a complete define-by-run **Automatic Differentiation (Autograd)** computational graph, ensuring mathematically guaranteed gradients for neural network backpropagation.
- Precision preserved elegantly across massive operations (`float64` strict type preservation).
- Full broadcasting semantics for operations (`add`, `matmul`, `pow`, `sum`).
- Topologically sorted backwards passes.

### 2. 🔌 `mayini.nn` (Deep Learning)
A rich PyTorch-like Neural Network library mapping seamlessly to the Tensor backend.
- **Layers:** `Linear`, `Conv2D`, `RNN`
- **Activations:** `ReLU`, `Sigmoid`, `Tanh`, `Softmax`
- **Optimizers:** `Adam`, `RMSprop`, `SGD` (with momentum)
- **Losses:** `MSELoss`, `CrossEntropyLoss`, `BCELoss`
- **Structure:** Standardized `Module` inheritance blocks with recursive parameter registration.

### 3. 🤖 `mayini.ml` (Classical Machine Learning)
Robust statistical and classical machine learning algorithms, optimized for clarity and speed.
- **Supervised:** Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Decision Trees, Naive Bayes, Linear & Logistic Regression.
- **Unsupervised:** K-Means Clustering, PCA (Principal Component Analysis), Isolation Forests (Anomaly Detection).
- **Ensembles:** Bagging, Boosting (AdaBoost/Gradient style), Voting implementations.

### 4. 🧬 `mayini.neat` (NeuroEvolution of Augmenting Topologies)
A custom, fully-featured implementation of **NEAT**. Instead of relying on gradient descent, NEAT evolves both the topological structure and weights of artificial neural networks using genetic algorithms.
- Dynamically mutate network structures (add nodes, add connections).
- **Speciation** to protect topological innovations from being wiped out early.
- Custom activation registries and highly-configurable fitness evaluators.

### 5. 🛠️ `mayini.preprocessing` (Multimodal Pipelines & Gradio Widget)
An enterprise-grade multimodal data preprocessor spanning **Text**, **Image**, **Audio**, and **Video**.
- **No-Code Interactive Widget:** Launch a Gradio-powered UI to visualize and build your preprocessing pipelines instantly! Run cleaning, transformations, and feature extractions directly from your browser.
- **Text:** TF-IDF, Stemming, Tokenization, Text Normalization.
- **Image:** Bilinear Resizing, Sobel Edge Detection, Datagen Augmentations.
- **Audio:** STFT Spectrograms, MFCC feature extraction, Pitch Shifting.

---

## 💻 Code Examples & Quick Start

### Example 1: Basic Autograd Calculus
```python
from mayini.tensor import Tensor

# Define variables with gradient tracking
x = Tensor([2.0], requires_grad=True)
y = Tensor([3.0], requires_grad=True)

# Build a computational graph
z = (x ** 2) + (x * y)

# Compute gradients
z.backward()

print(f"z = {z.data}")         # z = 10.0
print(f"dz/dx = {x.grad}")     # 2x + y = 7.0 
print(f"dz/dy = {y.grad}")     # x = 2.0
```

### Example 2: Training a Multilayer Perceptron (MLP)
```python
from mayini.tensor import Tensor
import mayini.nn as nn
import mayini.optim as optim

# 1. Define Model Architecture
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(out))

model = MLP()

# 2. Setup Optimizer & Loss Metric
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# 3. Create Dataset (XOR problem)
X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = Tensor([[0], [1], [1], [0]])

# 4. Training Loop
for epoch in range(200):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, Y)
    loss.backward()
    optimizer.step()

print(f"Final Model Loss: {loss.data}")
```

### Example 3: Training an SVM (Classical ML)
```python
import numpy as np
from mayini.ml.supervised.svm import SVM

# Generate random dummy data
X = np.array([[1, 2], [1, 3], [5, 6], [6, 5]])
y = np.array([1, 1, -1, -1])

# Initialize Support Vector Machine
model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
model.fit(X, y)

# Predict
predictions = model.predict(np.array([[2, 2], [5, 5]]))
print(f"Predictions: {predictions}")
```

### Example 4: Launching the Gradio Preprocessor UI
Tired of formatting complex data arrays blindly? Use the built-in Gradio widget to instantly mock your pipelines:
```python
from mayini.preprocessing.widget import launch_widget

# Automatically opens the local server at http://localhost:7860
launch_widget()
```

---

## 🛠️ Testing & Stability
Mayini relies on rigorous regression testing to ensure robust deployments. 
- The framework guarantees **100% test suite stability**.
- The `mayini.tensor` operations are validated against **scipy.optimize** finite difference algorithms to ensure mathematically flawless differentiation boundaries.

Run the test suite locally:
```bash
pytest -v test/
```

## 🤝 Contributing
Contributions are always welcome. To get started:
1. Fork the codebase and branch off `main`.
2. Ensure you add robust unittests (`pytest`) for any mathematical algorithm updates.
3. Submit a Pull Request.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full legal layout.
