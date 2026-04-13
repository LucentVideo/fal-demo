"""lucent-serverless CLI entry points.

These are registered as console scripts in pyproject.toml:
  - lucent-spawn     -> cli.spawn:main
  - lucent-terminate -> cli.terminate:main

They exist because Layer 2 has no control plane yet. Once Layer 3 ships
these become dev/debug utilities — the real pod lifecycle happens inside
the scheduler and reaper.
"""
