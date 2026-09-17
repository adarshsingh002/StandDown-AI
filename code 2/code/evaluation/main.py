#!/usr/bin/env python3
"""
HackerRank Orchestrate: Buy or Wait?
Evaluation runner proxy pointing to code.main
"""

import sys
import os
from pathlib import Path

# Add project code directory to sys.path
code_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(code_dir))

from main import main

if __name__ == "__main__":
    main()
