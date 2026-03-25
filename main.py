"""
main.py
─────────────────────────────────────────────────────────────────────────────
Entry point for the Smart Attendance System.

Usage:
    python main.py            # Train (if needed) then start recognition
    python main.py --train    # Force re-train the face encodings
    python main.py --summary  # Show today's attendance summary and exit

─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys
from pathlib import Path

# ── Make sure the project root is on sys.path so 'src' is importable ─────────
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train      import run_training, ENCODINGS_FILE
from src.recognize  import run_recognition
from src.attendance import get_today_summary, get_all_logs


BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║      SMART ATTENDANCE SYSTEM — Face Recognition CV       ║
║           with Real-time Duplicate Detection             ║
╚══════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Attendance System using Face Recognition",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Force re-train face encodings from the dataset folder.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print today's attendance summary and exit.",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="List all saved attendance log files.",
    )
    return parser.parse_args()


def main() -> None:
    print(BANNER)
    args = parse_args()

    # ── Show summary only ─────────────────────────────────────────────────
    if args.summary:
        get_today_summary()
        return

    # ── List log files ────────────────────────────────────────────────────
    if args.logs:
        logs = get_all_logs()
        if not logs:
            print("No attendance logs found yet.")
        else:
            print("Saved attendance logs:")
            for log in logs:
                print(f"  • {Path(log).name}")
        return

    # ── Training phase ────────────────────────────────────────────────────
    if args.train or not ENCODINGS_FILE.exists():
        if not args.train:
            print("[INFO] No encodings file found. Running training automatically…\n")
        run_training()

    # ── Recognition phase ─────────────────────────────────────────────────
    run_recognition()


if __name__ == "__main__":
    main()
