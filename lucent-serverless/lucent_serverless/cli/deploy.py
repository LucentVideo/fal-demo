"""lucent deploy <app.py> — register an App with the controller.

Imports the given file, finds the App subclass, and calls ls.deploy().

Usage:
  lucent deploy app.py
  lucent deploy examples/echo/app.py --controller-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys

from dotenv import load_dotenv

from .. import App, deploy


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lucent deploy",
        description="Register a lucent-serverless App with the controller.",
    )
    p.add_argument("file", help="path to the Python file containing an App subclass")
    p.add_argument("--controller-url", default=None, help="override LUCENT_CONTROLLER_URL")
    p.add_argument("--api-key", default=None, help="override LUCENT_API_KEY")
    return p


def _load_app_class(filepath: str) -> type[App]:
    """Import a .py file and return the App subclass it defines."""
    spec = importlib.util.spec_from_file_location("_lucent_user_app", filepath)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {filepath!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)

    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, App) and obj is not App and obj.__module__ == module.__name__
    ]
    if not candidates:
        raise SystemExit(f"no App subclass found in {filepath!r}")
    if len(candidates) > 1:
        names = ", ".join(c.__name__ for c in candidates)
        raise SystemExit(f"multiple App subclasses in {filepath!r}: {names} — use one per file")
    return candidates[0]


def main() -> int:
    load_dotenv()
    args = _parser().parse_args()

    cls = _load_app_class(args.file)
    try:
        deploy(cls, controller_url=args.controller_url, api_key=args.api_key)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
