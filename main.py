"""
VisionCut AI Cloud - Main Application Entry Point
AI-powered automatic software tutorial editor for Windows desktop.
"""

import sys
import argparse
from pathlib import Path


def run_gui():
    from PySide6.QtWidgets import QApplication
    from visioncut.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("VisionCut AI Cloud")
    app.setOrganizationName("VisionCut")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def run_end_to_end_demo(auto_open: bool = True):
    print("\n=======================================================")
    print("      VISIONCUT AI CLOUD - COMPLETE PIPELINE DEMO      ")
    print("=======================================================\n")
    from visioncut.tests.test_pipeline import run_complete_demonstration
    run_complete_demonstration(auto_open=auto_open)


def main():
    parser = argparse.ArgumentParser(
        description="VisionCut AI Cloud - Automatic Software Tutorial Editor"
    )
    parser.add_argument("--demo", action="store_true", help="Run the complete automated 1-minute test tutorial demo and generate visual preview")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open output folder/report after demo")
    parser.add_argument("--test", action="store_true", help="Run all unit and integration test suites")
    args = parser.parse_args()

    if args.demo:
        run_end_to_end_demo(auto_open=not args.no_open)
    elif args.test:
        from visioncut.tests.test_preprocessing import test_preprocessing
        from visioncut.tests.test_queue import test_queue
        from visioncut.tests.test_kdenlive import test_kdenlive_pipeline
        print("\n[SUITE 1/3] Testing Preprocessing...")
        test_preprocessing()
        print("\n[SUITE 2/3] Testing Google Drive Queue...")
        test_queue()
        print("\n[SUITE 3/3] Testing Kdenlive Generation & Validation...")
        test_kdenlive_pipeline()
        print("\nAll test suites passed successfully!")
    else:
        run_gui()


if __name__ == "__main__":
    main()

