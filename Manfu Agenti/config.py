# ─────────────────────────────────────────────
#  config.py  —  central settings for the project
# ─────────────────────────────────────────────

# ── LLM (Capgemini Generative Engine) ──────────
LLM_URL    = "https://your-capgemini-engine-url/v1"   # replace with your URL
LLM_MODEL  = "amazon.nova.lite"
LLM_API_KEY = "your-api-key-here"                     # replace with your key

# ── Simulator settings ──────────────────────────
MACHINES        = ["M1", "M2", "M3", "M4", "M5"]
TICK_INTERVAL   = 5          # seconds between each reading
BUFFER_SIZE     = 20         # readings kept in memory per machine

# ── Sensor baseline values (normal healthy machine) ──
BASELINE = {
    "temperature": 65.0,     # °C
    "vibration":   0.50,     # mm/s
    "pressure":    30.0,     # bar
}

# ── Gaussian noise sigma per sensor ─────────────
NOISE = {
    "temperature": 3.0,
    "vibration":   0.10,
    "pressure":    1.5,
}

# ── Drift rate per tick in degrading mode ────────
DRIFT = {
    "temperature": 0.8,      # °C per tick
    "vibration":   0.08,     # mm/s per tick
    "pressure":    0.35,     # bar per tick
}

# ── Critical state ranges ────────────────────────
CRITICAL_RANGE = {
    "temperature": (95.0,  135.0),
    "vibration":   (2.5,   5.0),
    "pressure":    (48.0,  62.0),
}

# ── State transition probabilities (per tick) ────
TRANSITION_PROB = {
    "normal_to_degrading": 0.02,    # 2% chance each tick
    "degrading_to_critical": 0.03,  # 3% chance each tick
    "spike_chance": 0.05,           # 5% micro-spike in normal mode
}

RECOVERY_TICKS = 6   # ticks machine stays in recovery before back to normal

# ── Health score thresholds ──────────────────────
THRESHOLDS = {
    "temperature": {"warning": 72.0,  "critical": 90.0},
    "vibration":   {"warning": 0.85,  "critical": 1.8},
    "pressure":    {"warning": 36.0,  "critical": 46.0},
}

# ── Agent settings ───────────────────────────────
AGENT_CHECK_INTERVAL = 10   # seconds between agent checks
MAX_AGENT_ITERATIONS = 8    # max ReAct loop steps
