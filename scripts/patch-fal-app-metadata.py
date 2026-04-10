"""Patch fal.app._print_python_packages for broken importlib metadata (metadata=None).

Seen on RunPod / mixed apt+pip images: fal 1.x list-comp crashes with:
  TypeError: 'NoneType' object is not subscriptable
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

# Match the fragile list comprehension; capture indentation (fal uses 4 spaces).
_PATTERN = re.compile(
    r"^(\s*)packages = \[f\"\{dist\.metadata\[\'Name\'\]\}==\{dist\.version\}\" "
    r"for dist in distributions\(\)\]",
    re.MULTILINE,
)

_REPLACEMENT = (
    r"\1packages = [f\"{dist.metadata['Name']}=={dist.version}\" "
    r"for dist in distributions() if dist.metadata is not None]"
)


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
    new_text, n = _PATTERN.subn(_REPLACEMENT, text, count=1)
    if n != 1:
        print("Could not find _print_python_packages line to patch", file=sys.stderr)
        sys.exit(1)
    path.write_text(new_text)
    print("patched", path)


if __name__ == "__main__":
    main()
