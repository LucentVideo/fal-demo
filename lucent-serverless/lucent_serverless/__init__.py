"""lucent-serverless: a small serverless realtime platform.

The user-facing surface is intentionally tiny: an `App` base class to
inherit from, and a `@realtime(path)` decorator to mark a method as a
websocket handler. Everything else lives in the runner.
"""


class App:
    """Base class for a lucent-serverless app.

    Subclass this, set the class attributes you care about, override
    `setup()` to load models, and decorate one or more methods with
    `@realtime("/path")`.
    """

    machine_type: str = "H100 80GB"
    image_ref: str = ""
    keep_alive: int = 300
    min_concurrency: int = 0
    max_concurrency: int = 4

    def setup(self) -> None:
        """Override to load models. Runs once per pod boot, before serving."""


def realtime(path: str):
    """Mark a method as a websocket endpoint mounted at `path`."""

    def decorator(fn):
        fn._lucent_realtime_path = path
        return fn

    return decorator


__all__ = ["App", "realtime"]
