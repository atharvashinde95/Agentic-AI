"""
utils.py — Utility helpers for the Predictive Maintenance Agent
"""

import os
import sys


def print_banner():
    """Print the startup banner."""
    print("""
╔══════════════════════════════════════════════════════╗
║   🤖  Autonomous Predictive Maintenance Agent        ║
║        Real-Time Sensor Monitoring System            ║
╚══════════════════════════════════════════════════════╝
  Press Ctrl+C to stop the agent at any time.
""")


def print_config(interval: float, max_cycles: int):
    """Print the current run configuration."""
    cycles_str = str(max_cycles) if max_cycles else "∞ (unlimited)"
    print(f"  ⚙️  Config: Interval = {interval}s | Max Cycles = {cycles_str}\n")


def get_log_path() -> str:
    """Return the path to the log file."""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "maintenance_log.txt")


def check_python_version():
    """Ensure Python 3.10+ for type hint compatibility."""
    if sys.version_info < (3, 10):
        print("⚠️  Warning: Python 3.10+ recommended for best compatibility.")
