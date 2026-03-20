"""
main.py — Entry Point for the Predictive Maintenance Agent
Run this file to start the autonomous agent system.
"""

import time
import sys
import os

# Add project directory to path so imports work from any location
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import get_sensor_reading
from agent import MaintenanceAgent
from utils import print_banner, print_config, get_log_path, check_python_version


# ─────────────────────────────────────────────
# Configuration (adjust as needed)
# ─────────────────────────────────────────────

CYCLE_INTERVAL_SECONDS = 2.0   # Time between sensor readings
MAX_CYCLES = 30                 # Set to None to run indefinitely


# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────

def main():
    check_python_version()
    print_banner()
    print_config(CYCLE_INTERVAL_SECONDS, MAX_CYCLES)
    print(f"  📄 Log file: {get_log_path()}\n")

    # Instantiate the single agent
    agent = MaintenanceAgent()

    cycle = 1
    try:
        while True:
            # Stop after MAX_CYCLES (if set)
            if MAX_CYCLES and cycle > MAX_CYCLES:
                print("\n✅ Reached maximum cycle limit. Agent stopping.\n")
                break

            # Generate simulated sensor data
            sensor_data = get_sensor_reading(cycle)

            # Hand data to the agent — it handles everything from here
            agent.run_cycle(sensor_data)

            # Wait before next reading
            time.sleep(CYCLE_INTERVAL_SECONDS)
            cycle += 1

    except KeyboardInterrupt:
        print("\n\n⛔ Agent stopped by user (Ctrl+C).\n")

    print(f"  📄 Full log saved to: {get_log_path()}\n")


if __name__ == "__main__":
    main()
