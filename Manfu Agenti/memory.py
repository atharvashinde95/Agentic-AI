# ─────────────────────────────────────────────
#  memory.py  —  agent short-term + long-term memory
# ─────────────────────────────────────────────

import threading
from datetime import datetime
from collections import deque


class AgentMemory:
    """
    Short-term : last N agent decisions per machine (avoids duplicate alerts)
    Long-term  : full action history log (what was done, when, outcome)
    """

    def __init__(self, short_term_size: int = 5):
        self._lock = threading.Lock()

        # short-term: last decisions per machine  {machine_id: deque of dicts}
        self._short_term: dict[str, deque] = {}

        # long-term: global ordered action log
        self._action_log: list[dict] = []

        self._short_term_size = short_term_size

    # ── Short-term helpers ───────────────────────

    def get_last_decision(self, machine_id: str) -> dict | None:
        """What did the agent last decide for this machine?"""
        with self._lock:
            history = self._short_term.get(machine_id)
            if history and len(history) > 0:
                return history[-1]
        return None

    def was_recently_alerted(self, machine_id: str, within_n: int = 3) -> bool:
        """
        Returns True if an alert was fired for this machine
        in the last `within_n` decisions — prevents duplicate alerts.
        """
        with self._lock:
            history = list(self._short_term.get(machine_id, []))
        recent = history[-within_n:]
        return any(d.get("action") in ("alert_sent", "maintenance_scheduled")
                   for d in recent)

    def record_decision(self, machine_id: str, decision: dict):
        """Save latest agent decision to short-term memory."""
        with self._lock:
            if machine_id not in self._short_term:
                self._short_term[machine_id] = deque(maxlen=self._short_term_size)
            self._short_term[machine_id].append({
                **decision,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    # ── Long-term helpers ────────────────────────

    def log_action(
        self,
        machine_id:  str,
        action:      str,
        severity:    str,
        reason:      str,
        job_id:      str | None = None,
        alert_id:    str | None = None,
        health_score: int | None = None,
    ):
        """Append one action to the permanent log."""
        entry = {
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "machine_id":   machine_id,
            "action":       action,
            "severity":     severity,
            "reason":       reason,
            "job_id":       job_id,
            "alert_id":     alert_id,
            "health_score": health_score,
        }
        with self._lock:
            self._action_log.append(entry)

    def get_action_log(self, machine_id: str | None = None) -> list[dict]:
        """Return full log, optionally filtered by machine."""
        with self._lock:
            log = list(self._action_log)
        if machine_id:
            log = [e for e in log if e["machine_id"] == machine_id]
        return log

    def get_recent_actions(self, n: int = 20) -> list[dict]:
        """Return the most recent n actions across all machines."""
        with self._lock:
            return list(self._action_log)[-n:]

    def clear_short_term(self, machine_id: str):
        """Reset short-term memory for a machine (after recovery)."""
        with self._lock:
            if machine_id in self._short_term:
                self._short_term[machine_id].clear()

    def summary(self) -> dict:
        """Quick stats for dashboard header."""
        with self._lock:
            log = list(self._action_log)
        return {
            "total_actions":   len(log),
            "alerts_sent":     sum(1 for e in log if e["action"] == "alert_sent"),
            "jobs_scheduled":  sum(1 for e in log if e["action"] == "maintenance_scheduled"),
            "machines_active": len(set(e["machine_id"] for e in log)),
        }


# ── Singleton — shared across all modules ────────
memory = AgentMemory()
