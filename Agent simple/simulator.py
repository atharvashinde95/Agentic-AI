"""
simulator.py
------------
Simulates real-time machine sensor data:
- Temperature (°C)
- Vibration (mm/s)
- Pressure (bar)

Includes gradual increase, random noise, and occasional sudden spikes.
"""

import random
import math
import time


class SensorSimulator:
    def __init__(self):
        # Base values for each sensor
        self.base_temp = 65.0       # Normal operating temperature (°C)
        self.base_vibration = 2.5   # Normal vibration level (mm/s)
        self.base_pressure = 4.5    # Normal pressure (bar)

        # Gradual drift accumulators (simulate aging / wear)
        self.temp_drift = 0.0
        self.vibration_drift = 0.0
        self.pressure_drift = 0.0

        # Tick counter — used to inject periodic spikes
        self.tick = 0

        # Spike flags — when True, sensor spikes for a few ticks
        self.temp_spike_remaining = 0
        self.vib_spike_remaining = 0
        self.pressure_spike_remaining = 0

    def _maybe_trigger_spike(self):
        """Randomly decide to start a spike event for one sensor."""
        if random.random() < 0.05:  # 5% chance per tick
            sensor = random.choice(["temp", "vibration", "pressure"])
            duration = random.randint(2, 5)
            if sensor == "temp":
                self.temp_spike_remaining = duration
            elif sensor == "vibration":
                self.vib_spike_remaining = duration
            else:
                self.pressure_spike_remaining = duration

    def _apply_gradual_drift(self):
        """Slowly increase sensor values over time to simulate machine wear."""
        # Every 20 ticks, increase drift slightly
        if self.tick % 20 == 0 and self.tick > 0:
            self.temp_drift += random.uniform(0.0, 0.8)
            self.vibration_drift += random.uniform(0.0, 0.15)
            self.pressure_drift += random.uniform(0.0, 0.05)

    def read(self):
        """
        Generate and return one sensor reading as a dict.
        Called once per simulation cycle.
        """
        self.tick += 1
        self._apply_gradual_drift()
        self._maybe_trigger_spike()

        # --- Temperature ---
        temp = self.base_temp + self.temp_drift
        temp += random.gauss(0, 1.5)          # random noise
        if self.temp_spike_remaining > 0:
            temp += random.uniform(18, 35)     # spike magnitude
            self.temp_spike_remaining -= 1

        # --- Vibration ---
        vibration = self.base_vibration + self.vibration_drift
        vibration += random.gauss(0, 0.3)
        if self.vib_spike_remaining > 0:
            vibration += random.uniform(4, 9)
            self.vib_spike_remaining -= 1

        # --- Pressure ---
        pressure = self.base_pressure + self.pressure_drift
        pressure += random.gauss(0, 0.2)
        if self.pressure_spike_remaining > 0:
            pressure += random.uniform(2, 5)
            self.pressure_spike_remaining -= 1

        # Clamp to realistic minimums
        temp = max(0.0, round(temp, 2))
        vibration = max(0.0, round(vibration, 2))
        pressure = max(0.0, round(pressure, 2))

        return {
            "tick": self.tick,
            "timestamp": time.time(),
            "temperature": temp,
            "vibration": vibration,
            "pressure": pressure,
        }

    def reset(self):
        """Reset the simulator to initial state."""
        self.__init__()
