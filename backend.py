from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
import uuid
import datetime

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["waste_management"]
users_collection = db["users"]
requests_collection = db["requests"]
vehicles_collection = db["vehicles"]

# Models
class User(BaseModel):
    username: str
    password: str
    address: Optional[str] = None
    contact: Optional[str] = None

class Location(BaseModel):
    latitude: float
    longitude: float

class WasteRequest(BaseModel):
    user: str
    location: Location
    waste_type: str
    status: str = "pending"
    created_at: Optional[str] = None

class VehicleUpdate(BaseModel):
    vehicle_id: str
    location: Location

class RequestUpdate(BaseModel):
    request_id: str
    status: str

# User Registration
@app.post("/register")
def register(user: User):
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="User already exists")
    users_collection.insert_one(user.dict())
    return {"message": "User registered successfully"}

# Submit Waste Pickup Request
@app.post("/request")
def submit_request(request: WasteRequest):
    # Add unique ID and timestamp
    request_data = request.dict()
    request_data["id"] = str(uuid.uuid4())
    request_data["created_at"] = datetime.datetime.now().isoformat()
    
    requests_collection.insert_one(request_data)
    return {"message": "Pickup request submitted", "request_id": request_data["id"]}

# Get All Requests
@app.get("/requests")
def get_requests(status: Optional[str] = None):
    query = {}
    if status:
        query["status"] = status
    
    results = list(requests_collection.find(query, {"_id": 0}))
    return results


@app.post("/complete_request")
def complete_request(update: RequestUpdate):
    result = requests_collection.update_one(
        {"id": update.request_id},
        {"$set": {"status": update.status}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Request not found")
    
    return {"message": f"Request marked as {update.status}"}


@app.post("/update_vehicle")
def update_vehicle(vehicle: VehicleUpdate):
    vehicles_collection.update_one(
        {"vehicle_id": vehicle.vehicle_id},
        {"$set": {
            "location": vehicle.location.dict(),
            "updated_at": datetime.datetime.now().isoformat()
        }},
        upsert=True
    )
    return {"message": "Vehicle location updated"}

# Get All Vehicles
@app.get("/vehicles")
def get_vehicles():
    return list(vehicles_collection.find({}, {"_id": 0}))

# Get a specific vehicle location
@app.get("/vehicle/{vehicle_id}")
def get_vehicle(vehicle_id: str):
    vehicle = vehicles_collection.find_one({"vehicle_id": vehicle_id}, {"_id": 0})
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return vehicle

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)