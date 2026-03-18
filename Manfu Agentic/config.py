# core/config.py
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── LLM ──────────────────────────────────────────────────────────────
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://your-capgemini-engine-url/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "your-api-key-here")
LLM_MODEL    = os.getenv("LLM_MODEL",    "amazon.nova.lite")

# ── Simulator ─────────────────────────────────────────────────────────
MACHINES             = ["M1", "M2", "M3", "M4", "M5"]
TICK_INTERVAL        = int(os.getenv("TICK_INTERVAL",        "5"))   # seconds
AGENT_CHECK_INTERVAL = int(os.getenv("AGENT_CHECK_INTERVAL", "30"))  # seconds
MAX_AGENT_ITERATIONS = 8
BUFFER_SIZE          = 20   # readings kept per machine in memory
RECOVERY_TICKS       = 6    # ticks machine stays in recovering state

# ── Sensor baselines (healthy machine) ───────────────────────────────
BASELINE = {
    "temperature": 65.0,   # °C
    "vibration":   0.50,   # mm/s
    "pressure":    30.0,   # bar
}

# ── Gaussian noise (natural sensor variation) ─────────────────────────
NOISE = {
    "temperature": 3.0,
    "vibration":   0.10,
    "pressure":    1.50,
}

# ── Drift per tick in degrading mode ──────────────────────────────────
DRIFT = {
    "temperature": 0.80,
    "vibration":   0.08,
    "pressure":    0.35,
}

# ── Critical state value ranges ───────────────────────────────────────
CRITICAL_RANGE = {
    "temperature": (95.0,  135.0),
    "vibration":   (2.50,  5.00),
    "pressure":    (48.0,  62.0),
}

# ── State transition probabilities (per tick) ─────────────────────────
TRANSITION = {
    "normal_to_degrading":   0.02,
    "degrading_to_critical": 0.03,
    "spike_chance":          0.05,
}

# ── Health score thresholds ───────────────────────────────────────────
THRESHOLDS = {
    "temperature": {"warning": 72.0,  "critical": 90.0},
    "vibration":   {"warning": 0.85,  "critical": 1.80},
    "pressure":    {"warning": 36.0,  "critical": 46.0},
}

# ── Machine states and sensor status labels ───────────────────────────
MACHINE_STATES  = ["normal", "degrading", "critical", "recovering"]
SENSOR_STATUSES = ["normal", "warning", "failure"]
