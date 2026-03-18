# simulator/simulator.py
import random
import time
import threading
from collections import deque
from datetime import datetime

from core.config import (
    MACHINES, BASELINE, NOISE, DRIFT, CRITICAL_RANGE,
    TRANSITION, RECOVERY_TICKS, BUFFER_SIZE,
    TICK_INTERVAL, THRESHOLDS,
)
from core.logger import error_logger


# ══════════════════════════════════════════════════════════
#  Machine  —  single machine state machine + sensor generator
# ══════════════════════════════════════════════════════════
class Machine:
    """
    Represents one physical machine.
    Manages state transitions and generates realistic
    correlated sensor readings every tick.

    States:  normal → degrading → critical → recovering → normal
    """

    def __init__(self, machine_id: str):
        self.machine_id    = machine_id
        self.state         = "normal"
        self.recovery_left = 0
        self.drift_steps   = 0

        # internal sensor values — evolve continuously across ticks
        self._temp      = BASELINE["temperature"] + random.gauss(0, 2.0)
        self._vibration = BASELINE["vibration"]   + random.gauss(0, 0.05)
        self._pressure  = BASELINE["pressure"]    + random.gauss(0, 1.0)

    # ── State transition ──────────────────────────────────────────────
    def _transition(self):
        if self.state == "normal":
            if random.random() < TRANSITION["normal_to_degrading"]:
                self.state       = "degrading"
                self.drift_steps = 0

        elif self.state == "degrading":
            if random.random() < TRANSITION["degrading_to_critical"]:
                self.state = "critical"

        elif self.state == "recovering":
            self.recovery_left -= 1
            if self.recovery_left <= 0:
                # fully recovered — reset internals to healthy baseline
                self.state      = "normal"
                self._temp      = BASELINE["temperature"] + random.gauss(0, 2.0)
                self._vibration = BASELINE["vibration"]   + random.gauss(0, 0.05)
                self._pressure  = BASELINE["pressure"]    + random.gauss(0, 1.0)

    # ── Sensor value generation ───────────────────────────────────────
    def _generate_values(self) -> dict:
        """
        Produce one realistic sensor reading.
        - Normal    : baseline + gaussian noise + occasional micro-spike
        - Degrading : gradual correlated drift upward each tick
        - Critical  : high unstable values with large noise
        - Recovering: exponential decay back toward baseline
        """

        if self.state == "normal":
            temp      = BASELINE["temperature"] + random.gauss(0, NOISE["temperature"])
            vibration = BASELINE["vibration"]   + random.gauss(0, NOISE["vibration"])
            pressure  = BASELINE["pressure"]    + random.gauss(0, NOISE["pressure"])

            # micro-spike — brief noise event, should NOT trigger an alert
            if random.random() < TRANSITION["spike_chance"]:
                temp      += random.uniform(4, 8)
                vibration += random.uniform(0.2, 0.4)

        elif self.state == "degrading":
            self.drift_steps += 1
            # correlated drift: temp rise pulls vibration and pressure up together
            self._temp      += DRIFT["temperature"] + random.gauss(0, NOISE["temperature"])
            self._vibration += DRIFT["vibration"]   + random.gauss(0, NOISE["vibration"])
            self._pressure  += DRIFT["pressure"]    + random.gauss(0, NOISE["pressure"])

            # clamp just below critical zone so degrading stays in warning territory
            self._temp      = min(self._temp,      CRITICAL_RANGE["temperature"][0] - 1.5)
            self._vibration = min(self._vibration, CRITICAL_RANGE["vibration"][0]   - 0.10)
            self._pressure  = min(self._pressure,  CRITICAL_RANGE["pressure"][0]    - 0.50)

            temp, vibration, pressure = self._temp, self._vibration, self._pressure

        elif self.state == "critical":
            temp      = random.uniform(*CRITICAL_RANGE["temperature"]) + random.gauss(0, 6.0)
            vibration = random.uniform(*CRITICAL_RANGE["vibration"])   + random.gauss(0, 0.30)
            pressure  = random.uniform(*CRITICAL_RANGE["pressure"])    + random.gauss(0, 3.0)

            # keep internals in sync for smooth recovery later
            self._temp, self._vibration, self._pressure = temp, vibration, pressure

        elif self.state == "recovering":
            # exponential decay toward baseline — feels natural on the chart
            self._temp      += (BASELINE["temperature"] - self._temp)      * 0.25
            self._vibration += (BASELINE["vibration"]   - self._vibration) * 0.25
            self._pressure  += (BASELINE["pressure"]    - self._pressure)  * 0.25

            temp      = self._temp      + random.gauss(0, NOISE["temperature"] * 0.5)
            vibration = self._vibration + random.gauss(0, NOISE["vibration"]   * 0.5)
            pressure  = self._pressure  + random.gauss(0, NOISE["pressure"]    * 0.5)

        else:
            temp, vibration, pressure = (
                BASELINE["temperature"],
                BASELINE["vibration"],
                BASELINE["pressure"],
            )

        # physical clamps — sensors cannot go outside real-world limits
        temp      = max(40.0, min(temp,      140.0))
        vibration = max(0.10, min(vibration,   6.0))
        pressure  = max(15.0, min(pressure,   70.0))

        return {
            "temperature": round(temp,      2),
            "vibration":   round(vibration, 3),
            "pressure":    round(pressure,  2),
        }

    # ── Derive status label from sensor values ────────────────────────
    @staticmethod
    def _derive_status(v: dict) -> str:
        """
        Status is computed from actual sensor numbers, NOT from internal state.
        This means the agent must figure out severity from numbers alone —
        exactly like a real system.
        """
        t = v["temperature"]
        b = v["vibration"]
        p = v["pressure"]

        if (t >= THRESHOLDS["temperature"]["critical"] or
                b >= THRESHOLDS["vibration"]["critical"] or
                p >= THRESHOLDS["pressure"]["critical"]):
            return "failure"

        if (t >= THRESHOLDS["temperature"]["warning"] or
                b >= THRESHOLDS["vibration"]["warning"] or
                p >= THRESHOLDS["pressure"]["warning"]):
            return "warning"

        return "normal"

    # ── Public tick method ────────────────────────────────────────────
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
            "state":       self.state,   # internal — used only for dashboard badge colour
        }

    def trigger_recovery(self):
        """Called by the agent after a maintenance action is taken."""
        self.state         = "recovering"
        self.recovery_left = RECOVERY_TICKS
        self.drift_steps   = 0


# ══════════════════════════════════════════════════════════
#  SensorSimulator  —  runs all machines in background thread
# ══════════════════════════════════════════════════════════
class SensorSimulator:
    """
    Starts a background thread that ticks all machines every TICK_INTERVAL seconds.
    Maintains a thread-safe rolling buffer of the last BUFFER_SIZE readings per machine.
    All other modules read from this buffer — never from a file or database.
    """

    def __init__(self):
        self._machines: dict[str, Machine] = {m: Machine(m) for m in MACHINES}
        self._buffers:  dict[str, deque]   = {
            m: deque(maxlen=BUFFER_SIZE) for m in MACHINES
        }
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run_loop(self):
        while self._running:
            try:
                with self._lock:
                    for mid, machine in self._machines.items():
                        reading = machine.tick()
                        self._buffers[mid].append(reading)
            except Exception as exc:
                error_logger.error("Simulator tick error: %s", exc)
            time.sleep(TICK_INTERVAL)

    # ── Buffer access (thread-safe) ───────────────────────────────────
    def get_readings(self, machine_id: str, n: int = 10) -> list[dict]:
        """Return last n readings for a machine."""
        with self._lock:
            buf = list(self._buffers.get(machine_id, []))
        return buf[-n:] if len(buf) >= n else buf

    def get_latest(self, machine_id: str) -> dict | None:
        """Return the single most recent reading for a machine."""
        readings = self.get_readings(machine_id, n=1)
        return readings[-1] if readings else None

    def get_all_latest(self) -> dict[str, dict | None]:
        """Return latest reading for every machine — used by dashboard fleet view."""
        return {m: self.get_latest(m) for m in MACHINES}

    def get_state(self, machine_id: str) -> str:
        """Return internal machine state — used only for dashboard badge colour."""
        with self._lock:
            return self._machines[machine_id].state

    def trigger_recovery(self, machine_id: str):
        """Agent calls this after a maintenance action to start recovery."""
        with self._lock:
            if machine_id in self._machines:
                self._machines[machine_id].trigger_recovery()


# ── Shared singleton — imported by tools and dashboard ───────────────
simulator = SensorSimulator()
