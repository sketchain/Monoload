"""Monoload: single-copy weight loading for ComfyUI.

Submodules that touch ComfyUI (loader, patcher, nodes, convert) import
`comfy.*` themselves; importing this package alone does not.
"""

__version__ = "0.1.0"
