"""
ACAI — run.py
Entry point. Run this file to launch the application.
"""

import sys
from PyQt5.QtWidgets import QApplication
from gui.app import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ACAI")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
