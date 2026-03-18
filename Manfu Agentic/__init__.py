# tools/__init__.py
from tools.sensor_tool      import get_sensor_reading
from tools.health_tool      import analyze_health
from tools.trend_tool       import detect_trend
from tools.maintenance_tool import schedule_maintenance
from tools.alert_tool       import send_alert

ALL_TOOLS = [
    get_sensor_reading,
    analyze_health,
    detect_trend,
    schedule_maintenance,
    send_alert,
]
