"""
main.py
-------
Console entry point — runs the LangChain Maintenance Agent without the UI.

Usage:
    python main.py

Press Ctrl+C to stop.
Each cycle prints the full ReAct chain (via verbose=True in AgentExecutor)
plus a clean summary line.
"""

import time
from simulator import SensorSimulator
from agent import MaintenanceAgent


def run():
    print("=" * 65)
    print("  🤖 LangChain Autonomous Predictive Maintenance Agent")
    print("  Mode  : Console (verbose ReAct chain)")
    print("  Press Ctrl+C to stop")
    print("=" * 65)

    simulator = SensorSimulator()
    agent     = MaintenanceAgent()

    try:
        while True:
            reading = simulator.read()
            print(f"\n{'═'*65}")
            print(f"  📡 Tick #{reading['tick']} — "
                  f"T={reading['temperature']}°C  "
                  f"V={reading['vibration']}mm/s  "
                  f"P={reading['pressure']}bar")
            print(f"{'═'*65}")

            result = agent.run_cycle(reading)

            print(f"\n  ✅ RESULT  |  Status: {result['status']}  |  "
                  f"Action: {result['action']}  |  Risk: {result['risk']}")
            print(f"  🔧 Tool   : {result['tool_used']}")
            if result["diagnoses"]:
                print(f"  🩺 Faults : {', '.join(result['diagnoses'][:2])}")
            print(f"  📋 Summary: {result['summary'][:120]}")

            time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n\n  Simulation stopped after {agent.cycle_count} cycles.")
        print("  Goodbye! 👋")


if __name__ == "__main__":
    run()
