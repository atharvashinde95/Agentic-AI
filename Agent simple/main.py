"""
main.py
-------
Entry point for running the agent in CONSOLE mode (no UI).

Usage:
    python main.py

Press Ctrl+C to stop.
"""

import time
from simulator import SensorSimulator
from agent import MaintenanceAgent


def run():
    print("=" * 60)
    print("  🤖 Autonomous Predictive Maintenance Agent")
    print("  Mode: Console")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    simulator = SensorSimulator()
    agent     = MaintenanceAgent()

    cycle = 0
    try:
        while True:
            cycle += 1
            reading = simulator.read()
            result  = agent.run_cycle(reading)

            # ── Print cycle summary ──────────────────────────────── #
            print(f"\n{'─'*55}")
            print(f"  Tick #{result['tick']:>4}  |  Status: {result['status']:<8}  |  Risk: {result['risk']}")
            print(f"{'─'*55}")
            print(f"  Temp     : {reading['temperature']:>6.2f} °C   [{result['sensor_status']['temperature']}]")
            print(f"  Vibration: {reading['vibration']:>6.2f} mm/s [{result['sensor_status']['vibration']}]")
            print(f"  Pressure : {reading['pressure']:>6.2f} bar  [{result['sensor_status']['pressure']}]")

            if result["diagnoses"]:
                print(f"  Diagnosis: {', '.join(result['diagnoses'])}")

            print(f"  Decision : {result['action']}")
            print(f"  Tool Used: {result['tool_result']['tool']}")
            print(f"  Confidence: {result['confidence']}")

            if result["llm_used"] and result["llm_explanation"]:
                print(f"\n  🤖 LLM Says: {result['llm_explanation']}")

            time.sleep(2)  # 2-second cycle

    except KeyboardInterrupt:
        print(f"\n\n  Simulation stopped after {cycle} cycles.")
        print(f"  LLM was called {agent.llm_call_count} time(s).")
        print("  Goodbye! 👋")


if __name__ == "__main__":
    run()
