"""
CellVision - Main Application Entry Point
Imports and runs the enhanced UI
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the enhanced UI
from ui.app_enhanced import main

if __name__ == "__main__":
    main()
