"""Monoload's ComfyUI nodes.

Each node is a class in a module of this package with the classic ComfyUI
node interface (INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY) plus
DISPLAY_NAME; list it in NODES and the plugin entry (../../__init__.py)
registers it under its class name. Nodes are registered whatever the
switches say (MONOLOAD_DISABLE included), so that a saved workflow that uses
one still loads; each node says itself when a switch makes it a no-op.
"""

from .vae_settings import MonoloadVAESettings

NODES = (MonoloadVAESettings,)

NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in NODES}
NODE_DISPLAY_NAME_MAPPINGS = {cls.__name__: cls.DISPLAY_NAME for cls in NODES}
