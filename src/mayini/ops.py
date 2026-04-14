import numpy as np
from .tensor import Tensor

def concatenate(tensors, axis=0):
    """Concatenate tensors along an axis."""
    data_list = [t.data if isinstance(t, Tensor) else t for t in tensors]
    result_data = np.concatenate(data_list, axis=axis)
    
    requires_grad = any(t.requires_grad for t in tensors if isinstance(t, Tensor))
    out = Tensor(result_data, requires_grad=requires_grad)
    out.op = f"ConcatBackward(axis={axis})"
    out.is_leaf = False
    out.prev = set(t for t in tensors if isinstance(t, Tensor))

    def _backward():
        if out.grad is not None:
            # Split grad back to children
            indices = np.cumsum([t.shape[axis] for t in tensors])[:-1]
            grads = np.split(out.grad, indices, axis=axis)
            for i, t in enumerate(tensors):
                if isinstance(t, Tensor) and t.requires_grad:
                    if t.grad is None:
                        t.grad = grads[i]
                    else:
                        t.grad = t.grad + grads[i]

    out._backward = _backward
    return out

def stack(tensors, axis=0):
    """Stack tensors along a new axis."""
    # Reshape each tensor to have a new dimension
    reshaped = [t.reshape((*t.shape[:axis], 1, *t.shape[axis:])) for t in tensors]
    return concatenate(reshaped, axis=axis)
