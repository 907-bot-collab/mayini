import numpy as np


def assert_arrays_close(arr1, arr2, rtol=1e-5, atol=1e-6):
    """Assert that two arrays are close in value."""
    # Handle mayini Tensors by converting to numpy
    # Avoid using .data on numpy arrays as it returns a buffer
    if hasattr(arr1, "data") and not isinstance(arr1, np.ndarray):
        arr1 = arr1.data
    if hasattr(arr2, "data") and not isinstance(arr2, np.ndarray):
        arr2 = arr2.data
    
    # If specifically comparing against None (e.g. grad is None)
    if arr1 is None or arr2 is None:
        assert arr1 is arr2, f"One is None, other is {type(arr1 if arr2 is None else arr2)}"
        return

    np.testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol)


# Aliases
assert_tensors_close = assert_arrays_close


def assert_gradient_close(t, expected_grad, rtol=1e-5, atol=1e-6):
    """Assert that a tensor's gradient matches expected value."""
    assert_arrays_close(t.grad, expected_grad, rtol=rtol, atol=atol)


def assert_shape_equal(arr, expected_shape):
    """Assert that array has expected shape."""
    if hasattr(arr, "shape"):
        shape = arr.shape
    else:
        shape = np.array(arr).shape
    assert shape == expected_shape, f"Expected shape {expected_shape}, got {shape}"


def numerical_gradient(func, x, h=1e-5):
    """Compute numerical gradient for testing automatic differentiation."""
    # Handle Tensor
    xr = x.data if hasattr(x, "data") else x
    grad = np.zeros_like(xr)
    it = np.nditer(xr, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        old_val = xr[idx].copy()

        # x is the tensor, func(x) should use the modified x.data
        xr[idx] = old_val + h
        pos_output = func(x)
        p_val = pos_output.item() if hasattr(pos_output, "item") else pos_output
        
        xr[idx] = old_val - h
        neg_output = func(x)
        n_val = neg_output.item() if hasattr(neg_output, "item") else neg_output
        
        grad[idx] = (p_val - n_val) / (2 * h)
        xr[idx] = old_val

        it.iternext()

    return grad
