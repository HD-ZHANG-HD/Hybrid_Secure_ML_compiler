"""Shared pytest fixtures and path setup.

`operator_execution_framework/` is the import root for the project; add it
to `sys.path` so tests can `import operators.*` / `import runtime.*`
without requiring the package to be installed.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
