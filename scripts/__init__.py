"""Utility scripts for development and analysis"""

import sys
import os

# Add project root to Python path for all scripts
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)