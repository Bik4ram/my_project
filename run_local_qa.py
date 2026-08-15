"""
Backward-compatibility shim.

The real CLI lives in ``semrag.cli`` and is installed as the
``semrag`` console script by ``pip install -e .``. This file is kept so
existing invocations like ``python run_local_qa.py index ...`` continue
to work.
"""
from __future__ import annotations

import sys

from semrag.cli import cmd_ask, cmd_index, main  # noqa: F401 — re-export

if __name__ == "__main__":
    sys.exit(main())
