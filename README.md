# Mayini Framework
![PyPI](https://img.shields.io/pypi/v/mayini-framework?color=blue&label=pypi%20version)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Test Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)

**Mayini** is a comprehensive, from-scratch Deep Learning and Machine Learning framework built in pure Python. Designed with a PyTorch-like API, it aims to demystify underlying AI mechanics while providing robust, high-performance features for deep neural networks, classical ML models, evolutionary algorithms, and multimodal data preprocessing.

---

## 📦 Installation

Install Mayini directly from PyPI:

```bash
pip install mayini-framework==0.8.0
```

---

## 🚀 Key Modules & Features

Mayini provides five carefully architected core pillars:

### 1. 🧠 `mayini.tensor` (Autograd Engine)
A custom multidimensional array computing engine built on top of NumPy. It features a complete define-by-run **Automatic Differentiation (Autograd)** computational graph, ensuring mathematically guaranteed gradients for neural network backpropagation.
- Precision preserved across massive operations (`float64` strict mode).
- Full broadcasting semantics for operations (`add`, `matmul`, `pow`, `sum`).

### 2. 🔌 `mayini.nn` (Deep Learning)
A rich PyTorch-like Neural Network library mapping seamlessly to the Tensor backend.
- Layers: `Linear`, `Conv2D`, `RNN`
- Activations: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`
- Optimizers: `Adam`, `RMSprop`, `SGD`
- Standardized `Module` inheritance blocks.

### 3. 🤖 `mayini.ml` (Classical Machine Learning)
Robust statistical and classical machine learning implementations.
- **Supervised:** Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Decision Trees, Naive Bayes, Linear & Logistic Regression.
- **Unsupervised:** K-Means Clustering, PCA (Decomposition), Isolation Forests (Anomaly Detection).
- **Ensemble Methods:** Bagging, Boosting (e.g., AdaBoost), Voting implementations.

### 4. 🧬 `mayini.neat` (NeuroEvolution)
A complete implementation of **NEAT (NeuroEvolution of Augmenting Topologies)**.
- Mutate network structures dynamically (nodes & connections).
- Speciation to protect topological innovations.
- Custom activation registries and highly-configurable fitness evaluators (e.g., `XORFitnessEvaluator`).

### 5. 🛠️ `mayini.preprocessing` (Multimodal Pipelines & Gradio Widget)
An enterprise-grade multimodal data preprocessor spanning **Text**, **Image**, **Audio**, and **Video**.
- **No-Code Interactive Widget:** Launch a Gradio-powered UI to visualize and build your preprocessing pipelines instantly! Run cleaning, transformations, and feature extractions directly from your browser.
  ```python
  from mayini.preprocessing.widget import launch_widget
  launch_widget()
  ```
- **Text:** TF-IDF, Stemming, Tokenization, Text Normalization.
- **Image:** Bilinear Resizing, Sobel Edge Detection, Datagen Augmentations.
- **Audio:** STFT Spectrograms, MFCC feature extraction, Pitch Shifting.

---

## 💻 Quick Start

### 1. Training a Neural Network
```python
from mayini.tensor import Tensor
import mayini.nn as nn
import mayini.optim as optim

# 1. Define Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

model = MLP()

# 2. Setup Optimizer & Loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# 3. Training Loop
x_data = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = Tensor([[0], [1], [1], [0]])

for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(x_data)
    loss = criterion(predictions, y_data)
    loss.backward()
    optimizer.step()

print(f"Final Loss: {loss.data}")
```

### 2. Launching the Preprocessor Widget
Need to test how a text snippet gets tokenized or how an image rotate operation affects its tensor shape?
```python
from mayini.preprocessing.widget import launch_widget

# Launch interactive UI at localhost:7860
launch_widget()
```

---

## 🛠️ Testing & Stability
Mayini framework heavily embraces rigorous testing.
- The framework guarantees **100% test coverage stability** (198/198 passing unittests!).
- Numerical Gradients validated strictly against finite-difference calculus methods to guarantee autograd mathematical precision.

## 🤝 Contributing
Contributions are always welcome. To get started:
1. Fork the repo and initialize your git branches.
2. Ensure you add robust unittests (`pytest`) for any new ML algorithms or backend Tensor operations.
3. Submit a Pull Request.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more.
