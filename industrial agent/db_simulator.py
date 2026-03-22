class DBSimulator:
    """
    Simulates a TimescaleDB / product recipe database.
    Each product has a per-batch material requirement in litres.
    In production, replace with a real DB client (psycopg2 / SQLAlchemy).
    """

    def __init__(self):
        self._recipes = {
            "product_a": {
                "product_name": "Product_A",
                "batch_size_litres": 200,
                "materials_per_batch": {
                    "tank_1": 100,
                    "tank_2": 60,
                    "tank_3": 40,
                },
                "required_machines": ["mixer_1", "reactor_1", "filler_1"],
                "steps": ["mix", "react", "fill"],
            },
            "product_b": {
                "product_name": "Product_B",
                "batch_size_litres": 150,
                "materials_per_batch": {
                    "tank_1": 50,
                    "tank_2": 70,
                    "tank_3": 30,
                },
                "required_machines": ["mixer_2", "reactor_2", "filler_2"],
                "steps": ["mix", "react", "fill"],
            },
            "product_c": {
                "product_name": "Product_C",
                "batch_size_litres": 300,
                "materials_per_batch": {
                    "tank_1": 120,
                    "tank_2": 100,
                    "tank_3": 80,
                },
                "required_machines": ["mixer_1", "reactor_1", "filler_1"],
                "steps": ["mix", "react", "fill"],
            },
        }

    def get_product(self, product_name: str) -> dict:
        key = product_name.lower().replace(" ", "_")
        if key in self._recipes:
            return self._recipes[key]
        return {"error": f"Product '{product_name}' not found. Available: {list(self._recipes.keys())}"}

    def list_products(self) -> list:
        return list(self._recipes.keys())


# Singleton instance — imported by tools.py
db_simulator = DBSimulator()
