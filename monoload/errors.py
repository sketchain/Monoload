"""Monoload exception types."""


class MonoloadError(RuntimeError):
    """Base class for all Monoload errors."""


class MonoloadUnsupportedError(MonoloadError):
    """A situation Monoload refuses instead of falling back to the native path
    that modifies weights in place and keeps a backup copy."""

    def __init__(self, kind, message, key=None):
        self.kind = kind
        self.key = key
        head = "[Monoload] 不支持（{}）".format(kind)
        if key is not None:
            head += " key={}".format(key)
        super().__init__("{}: {}".format(head, message))


class MonoloadVAEOOMError(MonoloadError):
    """A managed VAE decode ran out of memory with the smallest workspace.
    Monoload never falls back to the approximate tiled decode."""
