# Create the main conftest.py file with fixtures and common utilities
conftest_content = '''"""
Pytest configuration and common fixtures for mayini framework tests.
"""
import pytest
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import mayini components
import mayini as mn
from mayini.nn import *
from mayini.optim import *
from mayini.training import *

@pytest.fixture
def sample_tensor_2d():
    """Create a sample 2D tensor for testing."""
    return mn.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

@pytest.fixture
def sample_tensor_1d():
    """Create a sample 1D tensor for testing."""
    return mn.Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

@pytest.fixture
def sample_tensor_scalar():
    """Create a sample scalar tensor for testing."""
    return mn.Tensor(5.0, requires_grad=True)

@pytest.fixture
def sample_data_small():
    """Create small sample data for testing."""
    X = np.random.randn(10, 5).astype(np.float32)
    y = np.random.randint(0, 3, 10)
    return X, y

@pytest.fixture
def sample_data_medium():
    """Create medium sample data for testing."""
    X = np.random.randn(50, 10).astype(np.float32)
    y = np.random.randint(0, 5, 50)
    return X, y

@pytest.fixture
def simple_linear_model():
    """Create a simple linear model for testing."""
    return Sequential(
        Linear(10, 5),
        ReLU(),
        Linear(5, 3)
    )

@pytest.fixture
def simple_cnn_model():
    """Create a simple CNN model for testing."""
    return Sequential(
        Conv2D(1, 8, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2),
        Flatten(),
        Linear(8 * 14 * 14, 10)
    )

def assert_tensors_close(t1, t2, rtol=1e-5, atol=1e-6):
    """Assert that two tensors are close in value."""
    if hasattr(t1, 'data'):
        t1_data = t1.data
    else:
        t1_data = t1
    
    if hasattr(t2, 'data'):
        t2_data = t2.data
    else:
        t2_data = t2
        
    np.testing.assert_allclose(t1_data, t2_data, rtol=rtol, atol=atol)

def assert_gradient_close(tensor, expected_grad, rtol=1e-5, atol=1e-6):
    """Assert that tensor gradient is close to expected."""
    assert tensor.grad is not None, "Gradient should not be None"
    np.testing.assert_allclose(tensor.grad, expected_grad, rtol=rtol, atol=atol)

def numerical_gradient(func, x, h=1e-5):
    """Compute numerical gradient for testing automatic differentiation."""
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x.data[idx]
        
        x.data[idx] = old_value + h
        pos_output = func(x).item()
        
        x.data[idx] = old_value - h  
        neg_output = func(x).item()
        
        grad[idx] = (pos_output - neg_output) / (2 * h)
        x.data[idx] = old_value
        
        it.iternext()
    
    return grad
'''

# Save conftest.py
with open('conftest.py', 'w') as f:
    f.write(conftest_content)
    
print("Created conftest.py with common fixtures and utilities")
