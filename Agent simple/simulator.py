"""
simulator.py
------------
Simulates real-time industrial machine sensor data.

Sensors:
  - Temperature (°C)
  - Vibration   (mm/s)
  - Pressure    (bar)

Behaviour:
  - Gradual drift over time (simulates wear)
  - Gaussian noise every tick
  - Random spike events (2–5 ticks long, 5% chance per tick)
"""

import random
import time


class SensorSimulator:
    def __init__(self):
        # Baseline operating values
        self.base_temp      = 65.0
        self.base_vibration = 2.5
        self.base_pressure  = 4.5

        # Gradual drift accumulators
        self.temp_drift      = 0.0
        self.vibration_drift = 0.0
        self.pressure_drift  = 0.0

        self.tick = 0

        # Spike counters (how many ticks left in a spike)
        self.temp_spike_remaining     = 0
        self.vib_spike_remaining      = 0
        self.pressure_spike_remaining = 0

    def _maybe_trigger_spike(self):
        """5% chance per tick to start a random spike on one sensor."""
        if random.random() < 0.05:
            sensor   = random.choice(["temp", "vibration", "pressure"])
            duration = random.randint(2, 5)
            if sensor == "temp":
                self.temp_spike_remaining = duration
            elif sensor == "vibration":
                self.vib_spike_remaining = duration
            else:
                self.pressure_spike_remaining = duration

    def _apply_gradual_drift(self):
        """Every 20 ticks nudge drift upward to simulate machine ageing."""
        if self.tick % 20 == 0 and self.tick > 0:
            self.temp_drift      += random.uniform(0.0, 0.8)
            self.vibration_drift += random.uniform(0.0, 0.15)
            self.pressure_drift  += random.uniform(0.0, 0.05)

    def read(self) -> dict:
        """Produce one sensor snapshot dict."""
        self.tick += 1
        self._apply_gradual_drift()
        self._maybe_trigger_spike()

        # Temperature
        temp  = self.base_temp + self.temp_drift + random.gauss(0, 1.5)
        if self.temp_spike_remaining > 0:
            temp += random.uniform(18, 35)
            self.temp_spike_remaining -= 1

        # Vibration
        vib  = self.base_vibration + self.vibration_drift + random.gauss(0, 0.3)
        if self.vib_spike_remaining > 0:
            vib += random.uniform(4, 9)
            self.vib_spike_remaining -= 1

        # Pressure
        pres = self.base_pressure + self.pressure_drift + random.gauss(0, 0.2)
        if self.pressure_spike_remaining > 0:
            pres += random.uniform(2, 5)
            self.pressure_spike_remaining -= 1

        return {
            "tick"       : self.tick,
            "timestamp"  : time.time(),
            "temperature": max(0.0, round(temp, 2)),
            "vibration"  : max(0.0, round(vib,  2)),
            "pressure"   : max(0.0, round(pres, 2)),
        }

    def reset(self):
        self.__init__()
