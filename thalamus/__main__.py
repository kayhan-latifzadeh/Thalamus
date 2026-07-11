"""Lets you run the toolkit without installing it: ``python -m thalamus demo``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
