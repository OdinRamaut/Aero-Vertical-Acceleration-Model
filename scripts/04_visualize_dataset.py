"""
Script to launch the High-Performance Dataset Explorer (Qt).
"""
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication

# --- PATH SETUP ---
# Add project root to sys.path so we can import from 'src'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import logic from the modular GUI package
from src.gui.qt_explorer.main_window import DatasetExplorerWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern Look

    # Initialize Window
    # It will automatically point to DATASETS_DIR defined in config.py
    viewer = DatasetExplorerWindow()
    viewer.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()