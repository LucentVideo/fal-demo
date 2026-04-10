"""Patch fal.app._print_python_packages for broken importlib metadata (metadata=None).

Seen on RunPod / mixed apt+pip images: fal 1.x list-comp crashes with:
  TypeError: 'NoneType' object is not subscriptable
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

OLD = '    packages = [f"{dist.metadata[\'Name\']}=={dist.version}" for dist in distributions()]'
NEW = '    packages = [f"{dist.metadata[\'Name\']}=={dist.version}" for dist in distributions() if dist.metadata is not None]'


def main() -> None:
    spec = importlib.util.find_spec("fal")
    if not spec or not spec.origin:
        print("fal not found", file=sys.stderr)
        sys.exit(1)
    path = pathlib.Path(spec.origin).parent / "app.py"
    text = path.read_text()
    if "if dist.metadata is not None" in text and "_print_python_packages" in text:
        print("fal app.py already patched; skipping")
        return
    if OLD not in text:
        print("Could not find _print_python_packages line to patch", file=sys.stderr)
        print("Looking for:", repr(OLD), file=sys.stderr)
        sys.exit(1)
    path.write_text(text.replace(OLD, NEW, 1))
    print("patched", path)


if __name__ == "__main__":
    main()
