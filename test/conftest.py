# Create the main conftest.py file with fixtures and common utilities
conftest_content ='''"""
Pytest configuration and common fixtures for mayini framework tests.
"""
import pytest
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_tensor_2d():
    """Create a sample 2D numpy array for testing."""
    return np.array([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture
def sample_tensor_1d():
    """Create a sample 1D numpy array for testing."""
    return np.array([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def sample_tensor_scalar():
    """Create a sample scalar for testing."""
    return 5.0


@pytest.fixture
def sample_data_small():
    """Create small sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(10, 5).astype(np.float32)
    y = np.random.randint(0, 3, 10)
    return X, y


@pytest.fixture
def sample_data_medium():
    """Create medium sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(50, 10).astype(np.float32)
    y = np.random.randint(0, 5, 50)
    return X, y


@pytest.fixture
def blob_data():
    """Generate blob data for clustering tests."""
    try:
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=300, centers=3, random_state=42)
        return X, y
    except ImportError:
        # Fallback if sklearn not available
        np.random.seed(42)
        # Generate 3 clusters manually
        cluster1 = np.random.randn(100, 2) + np.array([0, 0])
        cluster2 = np.random.randn(100, 2) + np.array([5, 5])
        cluster3 = np.random.randn(100, 2) + np.array([5, 0])
        X = np.vstack([cluster1, cluster2, cluster3])
        y = np.array([0]*100 + [1]*100 + [2]*100)
        return X, y


@pytest.fixture
def regression_data():
    """Generate simple regression data."""
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X.squeeze() + 1 + np.random.randn(100) * 0.1
    return X, y


def assert_arrays_close(arr1, arr2, rtol=1e-5, atol=1e-6):
    """Assert that two arrays are close in value."""
    np.testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol)


def assert_shape_equal(arr, expected_shape):
    """Assert that array has expected shape."""
    assert arr.shape == expected_shape, f"Expected shape {expected_shape}, got {arr.shape}"


def numerical_gradient(func, x, h=1e-5):
    """Compute numerical gradient for testing automatic differentiation."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        x[idx] = old_value + h
        pos_output = func(x)
        
        x[idx] = old_value - h  
        neg_output = func(x)
        
        grad[idx] = (pos_output - neg_output) / (2 * h)
        x[idx] = old_value
        
        it.iternext()
    
    return grad
'''

# Save conftest.py
with open('conftest.py', 'w') as f:
    f.write(conftest_content)
    
print("Created conftest.py with common fixtures and utilities")
