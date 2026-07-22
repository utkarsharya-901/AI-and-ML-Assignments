from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="EcoTrace: Carbon Footprint Tracker API",
    description="Web-based application API to estimate annual carbon emissions based on lifestyle factors.",
    version="1.0.0"
)

# Input data schema for lifestyle factors
class LifestyleFactors(BaseModel):
    electricity_kwh: float
    transport_miles: float
    diet_multiplier: float = 1.0  # e.g., 1.0 for average, 0.8 for vegetarian, 1.5 for heavy meat

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Welcome to the EcoTrace API. Ready to track carbon footprints!",
        "docs_url": "/docs"
    }

@app.post("/estimate")
def estimate_carbon(factors: LifestyleFactors):
    # Basic estimation logic (emission factors are simplified for demonstration)
    electricity_emissions = factors.electricity_kwh * 0.92  # kg CO2 per kWh
    transport_emissions = factors.transport_miles * 0.411   # kg CO2 per mile
    
    # Calculate total and apply diet lifestyle multiplier
    total_emissions = (electricity_emissions + transport_emissions) * factors.diet_multiplier
    
    return {
        "lifestyle_factors_processed": True,
        "estimated_annual_carbon_kg": round(total_emissions, 2),
        "breakdown": {
            "electricity_kg": round(electricity_emissions, 2),
            "transport_kg": round(transport_emissions, 2)
        }
    }
