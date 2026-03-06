import os
import sys

# Ensure src is on the path for local runs
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mai.cli import main


if __name__ == "__main__":
    main()
