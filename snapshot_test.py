#!/usr/bin/env python3
"""
Command-line interface for the snapshot testing tool.

This script provides easy access to the snapshot testing functionality.
"""

import sys
from pathlib import Path

# Add the diff_tester module to the path
sys.path.insert(0, str(Path(__file__).parent))

from diff_tester.snapshot.cli import main

if __name__ == '__main__':
    sys.exit(main())
