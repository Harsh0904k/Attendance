"""
attendance.py
─────────────────────────────────────────────────────────────────────────────
Handles all CSV-based attendance operations:
  - Initialise / load the daily attendance log
  - Mark attendance for a recognised person
  - Duplicate detection (same name, same calendar day)
  - Summary reporting
─────────────────────────────────────────────────────────────────────────────
"""

import csv
import os
from datetime import datetime

# ── Constants ────────────────────────────────────────────────────────────────
ATTENDANCE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "attendance_logs")
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"
CSV_COLUMNS = ["Name", "Date", "Time", "Status"]


def _get_log_path(date_str: str) -> str:
    """Return the CSV file path for a given date string (YYYY-MM-DD)."""
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    return os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")


def _load_existing_records(log_path: str) -> dict[str, list[dict]]:
    """
    Load existing attendance records from a CSV file.
    Returns a dict mapping name → list of record dicts for fast duplicate lookup.
    """
    records: dict[str, list[dict]] = {}
    if not os.path.exists(log_path):
        return records

    with open(log_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row["Name"]
            records.setdefault(name, []).append(row)
    return records


def is_duplicate(name: str, date_str: str | None = None) -> bool:
    """
    Check whether *name* has already been marked present on *date_str*.
    If *date_str* is None the current calendar date is used.
    """
    if date_str is None:
        date_str = datetime.now().strftime(DATE_FORMAT)
    log_path = _get_log_path(date_str)
    records = _load_existing_records(log_path)
    return name in records


def mark_attendance(name: str) -> bool:
    """
    Attempt to mark attendance for *name*.

    Returns:
        True  – attendance recorded successfully.
        False – duplicate detected; attendance NOT recorded again.

    Side-effects:
        Appends a new row to today's CSV log (on success).
        Prints a status message to stdout in both cases.
    """
    now = datetime.now()
    date_str = now.strftime(DATE_FORMAT)
    time_str = now.strftime(TIME_FORMAT)
    log_path = _get_log_path(date_str)

    # ── Duplicate check ───────────────────────────────────────────────────
    if is_duplicate(name, date_str):
        print(
            f"[WARNING] Duplicate detected: '{name}' is already marked present "
            f"for {date_str}. Skipping."
        )
        return False

    # ── Write new record ──────────────────────────────────────────────────
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()          # write header only for new files
        writer.writerow({
            "Name": name,
            "Date": date_str,
            "Time": time_str,
            "Status": "Present",
        })

    print(f"[ATTENDANCE] ✓ '{name}' marked PRESENT at {time_str} on {date_str}.")
    return True


def get_today_summary() -> None:
    """Print a formatted summary of today's attendance to stdout."""
    date_str = datetime.now().strftime(DATE_FORMAT)
    log_path = _get_log_path(date_str)
    records = _load_existing_records(log_path)

    print("\n" + "=" * 50)
    print(f"  Attendance Summary — {date_str}")
    print("=" * 50)

    if not records:
        print("  No attendance recorded yet today.")
    else:
        for i, (name, entries) in enumerate(sorted(records.items()), start=1):
            entry = entries[0]          # first entry of the day
            print(f"  {i:>2}. {name:<25} {entry['Time']}")

    print(f"\n  Total present: {len(records)}")
    print("=" * 50 + "\n")


def get_all_logs() -> list[str]:
    """Return a sorted list of all existing attendance log file paths."""
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    files = sorted(
        f for f in os.listdir(ATTENDANCE_DIR) if f.endswith(".csv")
    )
    return [os.path.join(ATTENDANCE_DIR, f) for f in files]
