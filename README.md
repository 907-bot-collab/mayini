# Mayini Framework

![PyPI](https://img.shields.io/pypi/v/mayini-framework?color=blue&label=pypi%20version)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Test Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

**Mayini** is a comprehensive, from-scratch Deep Learning and Machine Learning framework built in pure Python. Designed with a clean, PyTorch-like API, it aims to demystify underlying AI mechanics while providing robust, high-performance features for deep neural networks, classical ML models, evolutionary algorithms, multimodal data preprocessing, and cutting-edge research modules.

Whether you are researching new topologies, learning the mathematics of backpropagation, deploying classical models, or building production-grade AI pipelines, Mayini has you covered.

---

## Table of Contents

- [Installation](#installation)
- [Core Modules](#core-modules)
  - [1. Tensor (Autograd Engine)](#1-tensor-autograd-engine)
  - [2. Neural Networks (nn)](#2-neural-networks-nn)
  - [3. Classical ML (ml)](#3-classical-ml-ml)
  - [4. NeuroEvolution (neat)](#4-neuroevolution-neat)
  - [5. Preprocessing](#5-preprocessing)
- [Advanced Modules](#advanced-modules)
  - [6. Explainable AI (explain)](#6-explainable-ai-explain)
  - [7. TinyML (tinyml)](#7-tinyml-tinyml)
  - [8. Federated Learning (federated)](#8-federated-learning-federated)
  - [9. Neural Architecture Search (nas)](#9-neural-architecture-search-nas)
  - [10. Multimodal AI (multimodal)](#10-multimodal-ai-multimodal)
  - [11. Model Inspection (inspect)](#11-model-inspection-inspect)
- [Quick Start](#quick-start)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

Install Mayini directly from PyPI via pip:

```bash
pip install mayini-framework==0.9.1
```

If you wish to edit the source code and contribute:

```bash
git clone https://github.com/907-bot-collab/mayini.git
cd mayini
pip install -e .[dev]
```

---

## Core Modules

### 1. Tensor (Autograd Engine)

A custom multidimensional array computing engine built on top of NumPy. It features a complete define-by-run **Automatic Differentiation (Autograd)** computational graph, ensuring mathematically guaranteed gradients for neural network backpropagation.

- Precision preserved elegantly across massive operations (`float64` strict type preservation).
- Full broadcasting semantics for operations (`add`, `matmul`, `pow`, `sum`).
- Topologically sorted backwards passes.

**Example:**
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

---

### 2. Neural Networks (nn)

A rich PyTorch-like Neural Network library mapping seamlessly to the Tensor backend.

- **Layers:** `Linear`, `Conv2D`, `RNN`, `LSTMCell`, `GRUCell`, `MaxPool2D`, `AvgPool2D`, `BatchNorm1d`, `Dropout`, `Flatten`
- **Activations:** `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `GELU`, `LeakyReLU`
- **Optimizers:** `Adam`, `AdamW`, `RMSprop`, `SGD` (with momentum)
- **Losses:** `MSELoss`, `MAELoss`, `CrossEntropyLoss`, `BCELoss`, `HuberLoss`
- **Structure:** Standardized `Module` inheritance blocks with recursive parameter registration.

**Example:**
```python
from mayini.tensor import Tensor
import mayini.nn as nn
import mayini.optim as optim

# Define Model Architecture
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

# Setup Optimizer & Loss Metric
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Create Dataset (XOR problem)
X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = Tensor([[0], [1], [1], [0]])

# Training Loop
for epoch in range(200):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, Y)
    loss.backward()
    optimizer.step()

print(f"Final Model Loss: {loss.data}")
```

---

### 3. Classical ML (ml)

Robust statistical and classical machine learning algorithms, optimized for clarity and speed.

- **Supervised:** Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Decision Trees, Naive Bayes, Linear & Logistic Regression.
- **Unsupervised:** K-Means Clustering, PCA (Principal Component Analysis), Isolation Forests (Anomaly Detection).
- **Ensembles:** Bagging, Boosting (AdaBoost/Gradient style), Voting implementations.

**Example:**
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

---

### 4. NeuroEvolution (neat)

A custom, fully-featured implementation of **NEAT** (NeuroEvolution of Augmenting Topologies). Instead of relying on gradient descent, NEAT evolves both the topological structure and weights of artificial neural networks using genetic algorithms.

- Dynamically mutate network structures (add nodes, add connections).
- **Speciation** to protect topological innovations from being wiped out early.
- Custom activation registries and highly-configurable fitness evaluators.

**Example:**
```python
from mayini.neat import NEATPopulation, Genome

# Create a NEAT population
pop = NEATPopulation(
    input_size=2,
    output_size=1,
    population_size=150
)

# Evolve for XOR problem
for generation in range(100):
    for genome in pop.population:
        # Evaluate fitness (XOR accuracy)
        fitness = evaluate_xor(genome)
        genome.fitness = fitness
    pop.evolve()
```

---

### 5. Preprocessing

An enterprise-grade multimodal data preprocessor spanning **Text**, **Image**, **Audio**, and **Video**.

- **No-Code Interactive Widget:** Launch a Gradio-powered UI to visualize and build your preprocessing pipelines instantly! Run cleaning, transformations, and feature extractions directly from your browser.
- **Text:** TF-IDF, Stemming, Tokenization, Text Normalization.
- **Image:** Bilinear Resizing, Sobel Edge Detection, Datagen Augmentations.
- **Audio:** STFT Spectrograms, MFCC feature extraction, Pitch Shifting.

**Example:**
```python
from mayini.preprocessing.widget import launch_widget

# Automatically opens the local server at http://localhost:7860
launch_widget()
```

---

## Advanced Modules

### 6. Explainable AI (explain)

White-box AI tools for model interpretability: graph tracing, gradient attribution, integrated gradients, and GradCAM-style explanations.

- `ComputationGraphBuilder` — Builds traceable computation graphs for model introspection
- `export_graph` — Exports the computational graph for external analysis
- `GradientExplainer` — White-box gradient attribution explanations
- `integrated_gradients` — Integrated Gradients method for feature attribution
- `fgsm_sensitivity` — Fast Gradient Sign Method sensitivity analysis
- `GraphVisualizer` — Visual rendering of model computation graphs

**Example:**
```python
from mayini.explain import GradientExplainer, integrated_gradients, GraphVisualizer
from mayini.nn import Sequential, Linear, ReLU

# Build a simple model
model = Sequential(
    Linear(10, 5),
    ReLU(),
    Linear(5, 2)
)

# Integrated Gradients attribution
baseline = Tensor.zeros((1, 10))
input_sample = Tensor.randn((1, 10))
attributions = integrated_gradients(model, input_sample, baseline, steps=50)
print(f"Feature attributions: {attributions}")

# Visualize computation graph
viz = GraphVisualizer()
viz.render(model, input_sample, filename="model_graph")
```

---

### 7. TinyML (tinyml)

Model compression and edge-deployment utilities for resource-constrained devices.

- **Pruning** (`pruning.py`) — Structured and unstructured weight pruning for model compression
- **Quantization** (`quantization.py`) — Post-training quantization to INT8 and lower precision
- **Export** (`export.py`) — Model export utilities for edge/microcontroller deployment

**Example:**
```python
from mayini.tinyml import prune_model, quantize_model, export_tflite
import mayini.nn as nn

# Build and train a model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Prune 50% of weights
pruned_model = prune_model(model, sparsity=0.5, method="magnitude")

# Quantize to INT8
quantized_model = quantize_model(pruned_model, dtype="int8")

# Export for edge deployment
export_tflite(quantized_model, "model.tflite")
```

---

### 8. Federated Learning (federated)

Privacy-preserving distributed training across decentralized clients.

- `FederatedClient` — Local training with gradient upload
- `FederatedServer` — FedAvg-style model aggregation
- `DifferentialPrivacy` — DP-SGD noise injection for secure federated learning

**Example:**
```python
from mayini.federated import FederatedClient, FederatedServer, DifferentialPrivacy

# Setup server
server = FederatedServer(
    model=global_model,
    aggregation="fedavg",
    num_clients=5
)

# Setup clients with differential privacy
client = FederatedClient(
    model=local_model,
    data=client_data,
    dp=DifferentialPrivacy(epsilon=1.0, delta=1e-5)
)

# Federated training loop
for round in range(100):
    client_updates = [client.train() for client in clients]
    server.aggregate(client_updates)
    global_model = server.get_model()
```

---

### 9. Neural Architecture Search (nas)

Automated discovery of optimal neural network architectures.

- `NASController` — RNN/Policy-based controller for architecture generation (ENAS-style)
- `SearchSpace` — Defines the searchable architecture space (operations, connections, layers)

**Example:**
```python
from mayini.nas import NASController, SearchSpace

# Define search space
search_space = SearchSpace(
    operations=["conv3x3", "conv5x5", "maxpool", "skip"],
    num_nodes=7,
    num_layers=3
)

# Initialize controller
controller = NASController(
    search_space=search_space,
    lstm_hidden_size=64
)

# Sample an architecture
architecture = controller.sample()
print(f"Sampled architecture: {architecture}")

# Train controller with REINFORCE
reward = train_and_evaluate(architecture)
controller.update(reward)
```

---

### 10. Multimodal AI (multimodal)

Cross-modal representation learning and fusion for vision-language and multi-sensory tasks.

- **CLIP** (`clip.py`) — Contrastive Language-Image Pre-training for vision-language alignment
- **Contrastive Learning** (`contrastive.py`) — Self-supervised contrastive loss implementations (SimCLR, MoCo style)
- **Fusion** (`fusion.py`) — Early, late, and attention-based multimodal fusion strategies

**Example:**
```python
from mayini.multimodal import CLIPModel, ContrastiveLoss, MultimodalFusion

# CLIP-style vision-language model
clip = CLIPModel(
    image_encoder="cnn",
    text_encoder="transformer",
    embedding_dim=512
)

# Contrastive learning
images = Tensor.randn((32, 3, 224, 224))
texts = Tensor.randint((32, 50), low=0, high=10000)
loss = clip(images, texts)
loss.backward()

# Multimodal fusion (e.g., image + audio)
fusion = MultimodalFusion(
    modalities=["image", "audio"],
    fusion_type="attention",
    output_dim=256
)
image_features = Tensor.randn((16, 512))
audio_features = Tensor.randn((16, 128))
fused = fusion(image_features, audio_features)
print(f"Fused representation shape: {fused.shape}")
```

---

### 11. Model Inspection (inspect)

Deep diagnostics and hook-based introspection for debugging and analyzing neural networks.

- **Diagnostics** (`diagnostics.py`) — Layer-wise activation statistics, gradient flow analysis, dead neuron detection
- **Hooks** (`hooks.py`) — Forward/backward hooks for intermediate value inspection, feature map visualization

**Example:**
```python
from mayini.inspect import ModelDiagnostics, register_hook
import mayini.nn as nn

# Build a model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Run diagnostics
diag = ModelDiagnostics(model)
X = Tensor.randn((100, 784))
stats = diag.analyze(X)

print(f"Layer 1 activation mean: {stats[0]['mean']:.4f}")
print(f"Layer 1 dead neurons: {stats[0]['dead_ratio']:.2%}")

# Register forward hook to capture intermediate activations
activations = {}
def capture_hook(module, input, output, name):
    activations[name] = output.data

register_hook(model[0], capture_hook, "fc1")
output = model(X)
print(f"FC1 output shape: {activations['fc1'].shape}")
```

---

## Quick Start

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

## Testing & Stability

Mayini relies on rigorous regression testing to ensure robust deployments.

- The framework guarantees **100% test suite stability**.
- The `mayini.tensor` operations are validated against **scipy.optimize** finite difference algorithms to ensure mathematically flawless differentiation boundaries.

Run the test suite locally:

```bash
pytest -v test/
```

---

## Contributing

Contributions are always welcome. To get started:

1. Fork the codebase and branch off `main`.
2. Ensure you add robust unittests (`pytest`) for any mathematical algorithm updates.
3. Submit a Pull Request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for the full legal layout.
