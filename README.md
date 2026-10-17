```python
import asyncio
from pathlib import Path
from intent_c.engine import AIEngine
from intent_c.sdk import BasePlugin, describe

# 1. Define your domain capability as a clean, isolated plugin
class HomeAutomationPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__()
        self.plugin_name = "climate"
        self.target_temperature = 19.0

    @describe("Changes the target room temperature of the climate controller")
    async def set_temperature(self, target_value: float) -> str:
        self.target_temperature = target_value
        return f"Climate target updated to {target_value}°C"

    def get_live_context(self) -> dict:
        """Exposes real-time system state for reactive conditional branching"""
        return {
            "current_temp": self.target_temperature,
            "heater_active": self.target_temperature < 22.0
        }

async def main():
    # 2. Instantiate the zero-overhead engine
    engine = AIEngine(ai_model="gemma4-e4b", routing_model="vector")
    await engine.initialize()
    
    # 3. Register your capabilities into RAM cached metadata
    engine.register_plugin(HomeAutomationPlugin())

    # 4. Compile and dispatch complex commands smoothly
    prompt = "If the heater is active (climate.heater_active), then turn the temperature to 23.5 degrees"
    results = await engine.run_engine(prompt)
    print(f"Execution Output: {results}")
    
    await engine.delete_engine()

if __name__ == "__main__":
    asyncio.run(main())
```
