"""Monoload exception types.

Every failure that would otherwise make Monoload silently diverge from native
ComfyUI behaviour is raised as one of these, with a message that says what
was wrong and what to do about it.
"""

RECONVERT_HINT = "请用 `python -m monoload.convert` 以当前的 ComfyUI 启动参数重新转换这个模型。"


class MonoloadError(RuntimeError):
    """Base class for all Monoload errors."""


class MonoloadFormatError(MonoloadError):
    """The converted file is unreadable, tampered with, or does not match what
    the current ComfyUI would build. Re-converting fixes it."""

    def __init__(self, path, problem, details=None):
        self.path = path
        self.problem = problem
        self.details = list(details or [])
        msg = "[Monoload] {}: {}".format(path, problem)
        if self.details:
            shown = self.details[:20]
            msg += "\n  - " + "\n  - ".join(str(d) for d in shown)
            if len(self.details) > len(shown):
                msg += "\n  - ...（共 {} 项）".format(len(self.details))
        msg += "\n" + RECONVERT_HINT
        super().__init__(msg)


class MonoloadUnsupportedError(MonoloadError):
    """A ComfyUI feature/mode that Monoload v1 deliberately refuses instead of
    falling back to a path that would hold a second copy of the weights."""

    def __init__(self, kind, message, key=None):
        self.kind = kind
        self.key = key
        head = "[Monoload] 不支持（{}）".format(kind)
        if key is not None:
            head += " key={}".format(key)
        super().__init__("{}: {}".format(head, message))
