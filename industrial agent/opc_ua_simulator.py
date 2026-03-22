class OPCUASimulator:
    """
    Simulates an OPC UA server for a batch plant.
    Mirrors the exact data shown in the plant SCADA screen:
      - 3 raw material tanks (fill % + litres)
      - Mixer (fill level %)
      - Pump (running %)
      - Reactor (fill level % + process readings)
      - Process sensors: PH, Pressure, Temperature, Energy
    In production, replace with a real asyncua client.
    """

    def __init__(self):

        # ── Raw material storage tanks ────────────────────────────────────
        self._tanks = {
            "tank_1": {
                "material":        "Material_1",
                "level_litres":    48491.438,
                "capacity_litres": 50000,
                "fill_percent":    97,
            },
            "tank_2": {
                "material":        "Material_2",
                "level_litres":    49316.95,
                "capacity_litres": 50000,
                "fill_percent":    99,
            },
            "tank_3": {
                "material":        "Material_3",
                "level_litres":    49192.086,
                "capacity_litres": 50000,
                "fill_percent":    98,
            },
        }

        # ── Mixer ─────────────────────────────────────────────────────────
        self._mixer = {
            "status":       "running",
            "fill_percent": 60,
            "description":  "Blends raw materials according to product recipe",
        }

        # ── Pump (between mixer and reactor) ──────────────────────────────
        self._pump = {
            "status":        "running",
            "speed_percent": 100,
            "description":   "Transfers blended mixture from mixer to reactor",
        }

        # ── Reactor ───────────────────────────────────────────────────────
        self._reactor = {
            "status":       "idle",
            "fill_percent": 0,
            "description":  "Chemical processing vessel",
        }

        # ── Process sensors (digital readouts shown in image) ─────────────
        self._process_sensors = {
            "ph":          {"value": 6.80,  "unit": "",      "label": "PH"},
            "pressure":    {"value": 1.10,  "unit": "Kpa",   "label": "Pressure"},
            "temperature": {"value": 24.70, "unit": "Deg C", "label": "Temperature"},
            "energy":      {"value": 0.10,  "unit": "kWh",   "label": "Energy"},
        }

    # ── Public getters ────────────────────────────────────────────────────────

    def get_tank_levels(self) -> dict:
        """Returns all tank fill levels and material info."""
        return self._tanks

    def get_machine_states(self) -> dict:
        """Returns operational status of mixer, pump, and reactor."""
        return {
            "mixer":   self._mixer,
            "pump":    self._pump,
            "reactor": self._reactor,
        }

    def get_process_sensors(self) -> dict:
        """Returns live sensor readings: PH, Pressure, Temperature, Energy."""
        return self._process_sensors

    def get_full_plant_status(self) -> dict:
        """Returns everything in one call — used by the sidebar."""
        return {
            "tanks":           self._tanks,
            "mixer":           self._mixer,
            "pump":            self._pump,
            "reactor":         self._reactor,
            "process_sensors": self._process_sensors,
        }


# Singleton instance — imported by tools.py and app.py
opc_ua_simulator = OPCUASimulator()
