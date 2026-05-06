"""
mayini.web — WebAssembly Deployment

Compiles Mayini models to WebAssembly text format (WAT) and generates
JavaScript wrapper stubs for in-browser inference.
"""

from .wasm_compiler import WASMCompiler

__all__ = ["WASMCompiler"]
