"""Local python package with the three perceptual model wrappers and the
frame-compositing logic. Shipped to the runner via
``local_python_modules = ["core"]`` on the fal.App subclass.

Mirrors the shape of ``fal_demos`` in upstream sana.py: a plain python
package with no fal imports inside, callable from both the deployed runner
and local smoke tests on a rented GPU.
"""


def detect_device() -> str:
    """Pick the best available accelerator: CUDA > MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
