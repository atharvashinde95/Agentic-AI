# ─────────────────────────────────────────────
#  simulator.py  —  realistic real-time sensor generator
# ─────────────────────────────────────────────

import random
import time
import threading
from datetime import datetime
from collections import deque
from config import (
    MACHINES, BASELINE, NOISE, DRIFT,
    CRITICAL_RANGE, TRANSITION_PROB,
    RECOVERY_TICKS, BUFFER_SIZE, TICK_INTERVAL, THRESHOLDS
)


class MachineSimulator:
    """Simulates one physical machine with realistic state transitions."""

    STATES = ["normal", "degrading", "critical", "recovering"]

    def __init__(self, machine_id: str):
        self.machine_id    = machine_id
        self.state         = "normal"
        self.recovery_left = 0
        self.drift_steps   = 0        # how many ticks in degrading mode

        # current internal sensor values (evolve over time)
        self.temp     = BASELINE["temperature"] + random.gauss(0, 2)
        self.vibration = BASELINE["vibration"]  + random.gauss(0, 0.05)
        self.pressure  = BASELINE["pressure"]   + random.gauss(0, 1)

    # ── State machine ────────────────────────────
    def _transition(self):
        """Decide whether to move to the next state this tick."""
        if self.state == "normal":
            if random.random() < TRANSITION_PROB["normal_to_degrading"]:
                self.state       = "degrading"
                self.drift_steps = 0

        elif self.state == "degrading":
            if random.random() < TRANSITION_PROB["degrading_to_critical"]:
                self.state = "critical"

        elif self.state == "recovering":
            self.recovery_left -= 1
            if self.recovery_left <= 0:
                self.state     = "normal"
                self.temp      = BASELINE["temperature"] + random.gauss(0, 2)
                self.vibration = BASELINE["vibration"]   + random.gauss(0, 0.05)
                self.pressure  = BASELINE["pressure"]    + random.gauss(0, 1)

    # ── Sensor value generator ───────────────────
    def _generate_values(self) -> dict:
        """Produce one realistic sensor reading based on current state."""

        if self.state == "normal":
            # healthy baseline + small gaussian noise
            temp      = BASELINE["temperature"] + random.gauss(0, NOISE["temperature"])
            vibration = BASELINE["vibration"]   + random.gauss(0, NOISE["vibration"])
            pressure  = BASELINE["pressure"]    + random.gauss(0, NOISE["pressure"])

            # occasional micro-spike (dust, power blip) — should NOT trigger alert
            if random.random() < TRANSITION_PROB["spike_chance"]:
                temp      += random.uniform(4, 8)
                vibration += random.uniform(0.2, 0.4)

        elif self.state == "degrading":
            self.drift_steps += 1
            # values drift upward gradually each tick — correlated sensors
            self.temp      += DRIFT["temperature"] + random.gauss(0, NOISE["temperature"])
            self.vibration += DRIFT["vibration"]   + random.gauss(0, NOISE["vibration"])
            self.pressure  += DRIFT["pressure"]    + random.gauss(0, NOISE["pressure"])

            # clamp so we don't exceed critical range during degrading
            self.temp      = min(self.temp,      CRITICAL_RANGE["temperature"][0] - 2)
            self.vibration = min(self.vibration, CRITICAL_RANGE["vibration"][0]   - 0.2)
            self.pressure  = min(self.pressure,  CRITICAL_RANGE["pressure"][0]    - 1)

            temp      = self.temp
            vibration = self.vibration
            pressure  = self.pressure

        elif self.state == "critical":
            # high unstable values with large noise
            temp      = random.uniform(*CRITICAL_RANGE["temperature"]) + random.gauss(0, 6)
            vibration = random.uniform(*CRITICAL_RANGE["vibration"])   + random.gauss(0, 0.3)
            pressure  = random.uniform(*CRITICAL_RANGE["pressure"])    + random.gauss(0, 3)

            self.temp      = temp
            self.vibration = vibration
            self.pressure  = pressure

        elif self.state == "recovering":
            # smoothly decay back toward baseline
            self.temp      += (BASELINE["temperature"] - self.temp)      * 0.25
            self.vibration += (BASELINE["vibration"]   - self.vibration) * 0.25
            self.pressure  += (BASELINE["pressure"]    - self.pressure)  * 0.25

            temp      = self.temp      + random.gauss(0, NOISE["temperature"] * 0.5)
            vibration = self.vibration + random.gauss(0, NOISE["vibration"]   * 0.5)
            pressure  = self.pressure  + random.gauss(0, NOISE["pressure"]    * 0.5)

        # absolute physical clamps — sensors can't go negative or impossibly high
        temp      = max(40.0,  min(temp,      140.0))
        vibration = max(0.1,   min(vibration,   6.0))
        pressure  = max(15.0,  min(pressure,   70.0))

        return {
            "temperature": round(temp,      2),
            "vibration":   round(vibration, 3),
            "pressure":    round(pressure,  2),
        }

    # ── Derive status label from sensor values ───
    @staticmethod
    def _derive_status(values: dict) -> str:
        """Status is derived from actual sensor numbers, not internal state."""
        t = values["temperature"]
        v = values["vibration"]
        p = values["pressure"]

        # critical if ANY sensor is in critical zone
        if (t >= THRESHOLDS["temperature"]["critical"] or
                v >= THRESHOLDS["vibration"]["critical"] or
                p >= THRESHOLDS["pressure"]["critical"]):
            return "failure"

        # warning if ANY sensor is in warning zone
        if (t >= THRESHOLDS["temperature"]["warning"] or
                v >= THRESHOLDS["vibration"]["warning"] or
                p >= THRESHOLDS["pressure"]["warning"]):
            return "warning"

        return "normal"

    # ── Produce one complete reading ─────────────
    def tick(self) -> dict:
        """Called every TICK_INTERVAL seconds. Returns one reading dict."""
        self._transition()
        values = self._generate_values()
        status = self._derive_status(values)

        return {
            "machine_id":  self.machine_id,
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": values["temperature"],
            "vibration":   values["vibration"],
            "pressure":    values["pressure"],
            "status":      status,
            "state":       self.state,   # internal — for dashboard colour only
        }

    def trigger_recovery(self):
        """Called by agent after maintenance action."""
        self.state         = "recovering"
        self.recovery_left = RECOVERY_TICKS
        self.drift_steps   = 0


# ── Multi-machine simulator with in-memory buffer ──
class SensorSimulator:
    """
    Runs all machines in a background thread.
    Exposes a thread-safe buffer of the last BUFFER_SIZE readings per machine.
    """

    def __init__(self):
        self.machines: dict[str, MachineSimulator] = {
            mid: MachineSimulator(mid) for mid in MACHINES
        }
        self.buffers: dict[str, deque] = {
            mid: deque(maxlen=BUFFER_SIZE) for mid in MACHINES
        }
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

    def start(self):
        """Start background ticker thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            with self._lock:
                for mid, machine in self.machines.items():
                    reading = machine.tick()
                    self.buffers[mid].append(reading)
            time.sleep(TICK_INTERVAL)

    def get_readings(self, machine_id: str, n: int = 10) -> list[dict]:
        """Return the last n readings for a machine (thread-safe)."""
        with self._lock:
            buf = list(self.buffers.get(machine_id, []))
        return buf[-n:] if len(buf) >= n else buf

    def get_latest(self, machine_id: str) -> dict | None:
        """Return the single most recent reading."""
        readings = self.get_readings(machine_id, 1)
        return readings[-1] if readings else None

    def get_all_latest(self) -> dict[str, dict]:
        """Return latest reading for every machine."""
        return {mid: self.get_latest(mid) for mid in MACHINES}

    def trigger_recovery(self, machine_id: str):
        """Agent calls this after a maintenance action."""
        with self._lock:
            if machine_id in self.machines:
                self.machines[machine_id].trigger_recovery()

    def get_state(self, machine_id: str) -> str:
        """Return current internal state (for dashboard badge colour)."""
        with self._lock:
            return self.machines[machine_id].state


# ── Singleton — shared across all modules ────────
simulator = SensorSimulator()
