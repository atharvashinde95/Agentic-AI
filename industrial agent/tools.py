import json
from langchain.tools import Tool

from opc_ua_simulator import opc_ua_simulator
from db_simulator import db_simulator


# ── Tool 1: Material availability ────────────────────────────────────────────

def material_availability_sync(_input: str = "") -> str:
    data = opc_ua_simulator.get_tank_levels()
    result = {}
    for tank_id, info in data.items():
        result[tank_id] = {
            "material":         info["material"],
            "available_litres": info["level_litres"],
            "capacity_litres":  info["capacity_litres"],
            "fill_percent":     info["fill_percent"],
        }
    return json.dumps(result, indent=2)

get_material_availability_tool = Tool(
    name="get_material_availability",
    func=material_availability_sync,
    description=(
        "Returns the current level (in litres) of raw-material tanks 1, 2, and 3 "
        "by reading their OPC-UA nodes on the batch-plant server. "
        "Use this to check if there is enough material to run production batches. "
        "No input required."
    ),
)


# ── Tool 2: Machine states ────────────────────────────────────────────────────

def machine_states_sync(_input: str = "") -> str:
    data = opc_ua_simulator.get_machine_states()
    result = {}
    for machine_id, info in data.items():
        entry = {
            "status":      info["status"],
            "description": info["description"],
            "operational": info["status"] == "running",
        }
        if "fill_percent" in info:
            entry["fill_percent"] = info["fill_percent"]
        if "speed_percent" in info:
            entry["speed_percent"] = info["speed_percent"]
        result[machine_id] = entry
    return json.dumps(result, indent=2)

get_machine_states_tool = Tool(
    name="get_machine_states",
    func=machine_states_sync,
    description=(
        "Returns the current operational status of all machines in the batch plant: "
        "mixer (with fill level %), pump (with speed %), and reactor (with fill level %). "
        "Use this to check if required equipment is available for a production run. "
        "No input required."
    ),
)


# ── Tool 3: Product recipe details ───────────────────────────────────────────

def get_product_details(product_name: str) -> str:
    data = db_simulator.get_product(product_name)
    return json.dumps(data, indent=2)

get_product_details_tool = Tool(
    name="get_product_details",
    func=get_product_details,
    description=(
        "Get the recipe details for a specific product including the quantity of each "
        "raw material required per batch (in litres) and which machines are needed. "
        "Input should be the product name as a string, e.g. 'Product_A', 'Product_B', or 'Product_C'."
    ),
)


# ── Tool 4: Process sensor readings ──────────────────────────────────────────

def get_process_sensors_sync(_input: str = "") -> str:
    data = opc_ua_simulator.get_process_sensors()
    result = {}
    for sensor_id, info in data.items():
        result[info["label"]] = f"{info['value']} {info['unit']}".strip()
    return json.dumps(result, indent=2)

get_process_sensors_tool = Tool(
    name="get_process_sensors",
    func=get_process_sensors_sync,
    description=(
        "Returns live process sensor readings from the batch plant: "
        "PH level, Pressure in Kpa, Temperature in Deg C, and Energy in kWh. "
        "Use this when asked about plant conditions or process health. "
        "No input required."
    ),
)


# ── Exported list of all tools ────────────────────────────────────────────────

all_tools = [
    get_material_availability_tool,
    get_machine_states_tool,
    get_product_details_tool,
    get_process_sensors_tool,
]
