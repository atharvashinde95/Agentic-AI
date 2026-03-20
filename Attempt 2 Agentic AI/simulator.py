"""
simulator.py — Sensor Data Generator
Simulates real-time machine sensor readings with gradual wear, noise, and spikes.
"""

import random
import math


# Base values for a healthy machine
BASE_TEMPERATURE = 70.0   # Celsius
BASE_VIBRATION = 0.5      # mm/s
BASE_PRESSURE = 100.0     # PSI

# Wear factor increases over time to simulate aging machine
_wear_factor = 0.0


def reset_wear():
    """Reset wear simulation (useful for testing)."""
    global _wear_factor
    _wear_factor = 0.0


def get_sensor_reading(cycle: int) -> dict:
    """
    Generate one sensor reading for the current cycle.

    Args:
        cycle: Current cycle number (used to simulate gradual wear)

    Returns:
        Dictionary with temperature, vibration, pressure values
    """
    global _wear_factor

    # Gradual wear: machine degrades slowly over time
    _wear_factor = cycle * 0.05  # Increases 0.05 per cycle

    # --- Temperature ---
    temp_wear = _wear_factor * 0.8           # Steady rise due to wear
    temp_noise = random.uniform(-2.0, 2.0)   # Normal fluctuation
    temp_spike = _random_spike(probability=0.08, magnitude=20.0)  # Occasional spike
    temperature = BASE_TEMPERATURE + temp_wear + temp_noise + temp_spike

    # --- Vibration ---
    vib_wear = _wear_factor * 0.03
    vib_noise = random.uniform(-0.05, 0.05)
    vib_spike = _random_spike(probability=0.06, magnitude=0.8)
    vibration = BASE_VIBRATION + vib_wear + vib_noise + vib_spike
    vibration = max(0.0, vibration)  # Vibration can't be negative

    # --- Pressure ---
    # Pressure oscillates slightly (like a pump cycle)
    pressure_wave = math.sin(cycle * 0.3) * 3.0
    pressure_noise = random.uniform(-2.0, 2.0)
    pressure_drop = _random_spike(probability=0.05, magnitude=-15.0)  # Sudden drop = leak
    pressure = BASE_PRESSURE + pressure_wave + pressure_noise + pressure_drop

    return {
        "temperature": round(temperature, 2),
        "vibration": round(vibration, 3),
        "pressure": round(pressure, 2),
        "cycle": cycle,
    }


def _random_spike(probability: float, magnitude: float) -> float:
    """
    Randomly generate a spike event.

    Args:
        probability: Chance (0–1) of a spike happening this cycle
        magnitude: Size of the spike (positive = high spike, negative = drop)

    Returns:
        Spike value or 0.0
    """
    if random.random() < probability:
        # Spike size varies between 60%–100% of max magnitude
        return magnitude * random.uniform(0.6, 1.0)
    return 0.0
