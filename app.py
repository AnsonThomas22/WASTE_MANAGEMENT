import streamlit as st
import requests
import time
import folium
import numpy as np
from PIL import Image
import random
from ultralytics import YOLO
import cv2
from geopy.distance import geodesic
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from streamlit_js_eval import streamlit_js_eval

# First, make sure streamlit_folium is imported correctly
try:
    from streamlit_folium import st_folium
except ImportError:
    st.error("Missing required package. Please install using: pip install streamlit-folium")
    st.stop()

# API Endpoint
API_BASE_URL = "http://127.0.0.1:8000"

# SG Palya Coordinates (Map centered here)
SG_PALYA_CENTER = (12.9352, 77.6095)

# Define specific roads in SG Palya for the truck route
SG_PALYA_ROUTE = [
    (12.9352, 77.6095),  # SG Palya Main Road Start
    (12.9360, 77.6105),  # Junction 1
    (12.9375, 77.6115),  # Junction 2
    (12.9385, 77.6095),  # Side Road 1
    (12.9370, 77.6085),  # Back to Main Road
    (12.9345, 77.6075),  # Junction 3
    (12.9335, 77.6090),  # Side Road 2
    (12.9352, 77.6095),  # Back to Start
]

# Global variables - use session state to persist between reruns
# Ensure session state variables exist
if 'vehicle_location' not in st.session_state:
    st.session_state.vehicle_location = SG_PALYA_ROUTE[0]

if 'active_requests' not in st.session_state:
    st.session_state.active_requests = []

if 'completed_requests' not in st.session_state:   # ✅ Track completed requests
    st.session_state.completed_requests = []

if 'current_route_index' not in st.session_state:
    st.session_state.current_route_index = 0

if 'route_progress' not in st.session_state:
    st.session_state.route_progress = 0

if 'notification_shown' not in st.session_state:
    st.session_state.notification_shown = False

if 'map_clicked_location' not in st.session_state:
    st.session_state.map_clicked_location = None

if 'animation_running' not in st.session_state:
    st.session_state.animation_running = False

if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()

if 'gps_location' not in st.session_state:
    st.session_state.gps_location = None

# Load YOLOv8 Model
@st.cache_resource
def load_custom_model():
    model_path = r"weights/best.pt"  # Replace with your actual model path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return YOLO(model_path)

try:
    with st.spinner("Loading AI model..."):
        model = load_custom_model()
except FileNotFoundError:
    st.warning("Model file not found. Using mock detection for demonstration.")
    model = None

# Custom class mapping for waste types
WASTE_CLASSES = {
    0: "Biodegradable",
    1: "Cardboard",
    2: "Glass",
    3: "Metal",
    4: "Paper",
    5: "Plastic"
}

def preprocess_image(image):
    """Enhance and preprocess the image while keeping color intact."""
    # Convert PIL image to OpenCV format (NumPy array)
    image = np.array(image)

    # Resize image to standard input size (YOLO prefers fixed sizes)
    image = cv2.resize(image, (640, 640))

    # Sharpen the image using a kernel
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  
    image = cv2.filter2D(image, -1, sharpening_kernel)

    # Convert back to PIL for brightness & contrast adjustments
    image_pil = Image.fromarray(image)

    # Enhance brightness & contrast
    enhancer = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer.enhance(1.2)  # Increase contrast slightly

    # Convert back to OpenCV format
    enhanced_image = np.array(image_pil)

    # Apply slight Gaussian blur to remove noise
    enhanced_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)

    return enhanced_image

def detect_waste(image):
    """Run YOLOv8 on the uploaded image and classify waste types with preprocessing."""
    if model is None:
        # If no model is loaded, return random detection for demo
        waste_types = list(WASTE_CLASSES.values())
        detections = [{"waste_type": random.choice(waste_types), "confidence": random.uniform(0.7, 0.95)}]
        return np.array(image), detections

    # Preprocess the image for better detection (without grayscale)
    processed_image = preprocess_image(image)

    # Run detection (Use .predict() in YOLOv8)
    results = model.predict(processed_image, conf=0.2)

    # Draw bounding boxes and annotations
    annotated_image = results[0].plot()

    # Extract detected objects
    detections = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])  # Confidence score
            class_id = int(box.cls[0])  # Class ID

            # Get waste category based on trained class names
            waste_type = WASTE_CLASSES.get(class_id, "Unknown")

            detections.append({
                "waste_type": waste_type,
                "confidence": confidence
            })

    return annotated_image, detections

def get_active_requests():
    """Fetch active pickup requests from the API"""
    try:
        with st.spinner("Fetching active requests..."):
            response = requests.get(f"{API_BASE_URL}/requests")
            if response.status_code == 200:
                return response.json()
            else:
                return []
    except:
        # Return some dummy data for demo purposes
        return [
            {"user": "user1", "location": {"latitude": 12.9360, "longitude": 77.6100}, "status": "pending", "waste_type": "Plastic"},
            {"user": "user2", "location": {"latitude": 12.9370, "longitude": 77.6090}, "status": "pending", "waste_type": "Bio-waste"}
        ]

def interpolate_position(point1, point2, progress):
    """Interpolate between two points for smooth movement"""
    lat = point1[0] + (point2[0] - point1[0]) * progress
    lng = point1[1] + (point2[1] - point1[1]) * progress
    return (lat, lng)

def update_vehicle_location():
    """Update the truck location every 5 seconds along the predefined route"""
    # Ensure session variables are initialized
    if "current_route_index" not in st.session_state:
        st.session_state.current_route_index = 0
        st.session_state.route_progress = 0.0
        st.session_state.last_update_time = time.time()  # Track last update time

    # Check if 5 seconds have passed
    if time.time() - st.session_state.last_update_time < 5:
        return  # Skip update if not enough time has passed

    # Update last update time
    st.session_state.last_update_time = time.time()

    # Get current and next route indices
    current_idx = st.session_state.current_route_index
    next_idx = (current_idx + 1) % len(SG_PALYA_ROUTE)

    # Move progress by a small amount
    st.session_state.route_progress += 0.1  # Adjust step size for smoother movement

    # If progress reaches 1, move to the next segment
    if st.session_state.route_progress >= 1.0:
        st.session_state.current_route_index = next_idx
        st.session_state.route_progress = 0.0
        current_idx = next_idx
        next_idx = (current_idx + 1) % len(SG_PALYA_ROUTE)

    # Compute new interpolated position
    current_point = SG_PALYA_ROUTE[current_idx]
    next_point = SG_PALYA_ROUTE[next_idx]
    st.session_state.vehicle_location = interpolate_position(
        current_point, next_point, st.session_state.route_progress
    )

    # Send updated location to backend
    try:
        requests.post(
            f"{API_BASE_URL}/update_vehicle",
            json={
                "vehicle_id": "truck-001",
                "location": {
                    "latitude": st.session_state.vehicle_location[0],
                    "longitude": st.session_state.vehicle_location[1]
                }
            }
        )
    except:
        pass  # Handle API failures gracefully

def check_nearby_requests():
    """Check if any requests are near the current vehicle location"""
    if "vehicle_location" not in st.session_state:
        st.error("🚨 Vehicle location not available!")
        return

    completed_requests = []
    remaining_requests = []  # New list to store remaining requests

    for request in st.session_state.active_requests:
        # Get request location
        request_location = (
            request.get("location", {}).get("latitude", 0),
            request.get("location", {}).get("longitude", 0)
        )

        try:
            # Calculate distance between vehicle and request location
            distance = geodesic(st.session_state.vehicle_location, request_location).meters

            if distance < 100: #changed from 50 to 100
                # Send completion request to API
                try:
                    response = requests.post(f"{API_BASE_URL}/complete_request",
                                             json={"request_id": request.get("id", ""), "status": "completed"})
                    response.raise_for_status()
                except requests.RequestException as e:
                    st.warning(f"⚠️ Error completing request: {e}")
                    remaining_requests.append(request)  # Keep request if API call fails
                    continue

                completed_requests.append(request)
                st.success(f"🎉 Pickup completed for {request.get('waste_type', 'waste')} at location {request_location[0]:.4f}, {request_location[1]:.4f}")
            else:
                remaining_requests.append(request)  # Keep request if not completed
        except Exception as e:
            st.warning(f"⚠️ Distance calculation failed: {e}")
            remaining_requests.append(request)  # Keep request if error occurs

    # Update session state with remaining requests
    st.session_state.active_requests = remaining_requests

def check_user_proximity():
    """Check if vehicle is near user location and show notification"""
    # Get user location (for demo, using SG Palya coordinates)
    user_location = SG_PALYA_CENTER

    # Calculate distance from vehicle
    distance = geodesic(st.session_state.vehicle_location, user_location).meters

    # If vehicle is within 100 meters and notification not shown
    if distance < 100 and not st.session_state.notification_shown:
        st.warning("🚛 Waste collection truck is approaching your location! Please bring your trash for collection.", icon="🔔")
        st.session_state.notification_shown = True
    elif distance >= 100:
        st.session_state.notification_shown = False

def get_current_truck_location():
    route_idx = st.session_state.current_route_index
    route_progress = st.session_state.route_progress

    if route_idx < len(SG_PALYA_ROUTE) - 1:
        start_lat, start_lon = SG_PALYA_ROUTE[route_idx]
        end_lat, end_lon = SG_PALYA_ROUTE[route_idx + 1]

        lat = start_lat + (end_lat - start_lat) * (route_progress / 100.0)
        lon = start_lon + (end_lon - start_lon) * (route_progress / 100.0)
        return lat, lon
    else:
        return SG_PALYA_ROUTE[-1]
    
def distance(lat1, lon1, lat2, lon2):
    # Simple distance calculation (replace with more accurate method if needed)
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 100000 #Approximate distance

def create_map():
    """Create and return a folium map centered on SG Palya."""
    # Create map centered on SG Palya
    m = folium.Map(location=SG_PALYA_CENTER, zoom_start=16)
    
    # Add truck marker
    folium.Marker(
        st.session_state.vehicle_location,
        tooltip="🚛 Waste Collection Truck",
        icon=folium.Icon(color='blue', icon='truck', prefix='fa')
    ).add_to(m)
    
    # Add route line
    folium.PolyLine(
        SG_PALYA_ROUTE,
        color='blue',
        weight=3,
        opacity=0.8
    ).add_to(m)
    
    # Check and remove completed requests
    updated_requests = []
    truck_location = st.session_state.vehicle_location
    
    # Add request markers
    for request in st.session_state.active_requests:
        request_location = (
            request.get("location", {}).get("latitude", 12.9352 + random.uniform(-0.003, 0.003)),
            request.get("location", {}).get("longitude", 77.6095 + random.uniform(-0.003, 0.003))
        )

        waste_type = request.get("waste_type", "Unknown")

        # Determine marker color based on waste type
        color_map = {
            "Bio-waste": "green",
            "Paper/Cardboard": "blue",
            "Plastic": "red",
            "Glass": "purple",
            "Medical waste": "orange"
        }
        color = color_map.get(waste_type, "gray")

        # Calculate distance between truck and request
        distance = geodesic(truck_location, request_location).meters
        
        if distance < 50:  # If truck is within 50 meters, mark request as completed
            st.success(f"✅ Pickup request completed at {request_location}!")
        else:
            updated_requests.append(request)  # Keep pending requests
            
            folium.Marker(
                request_location,
                tooltip=f"🗑️ Pickup Request: {waste_type} ({round(distance, 2)}m away)",
                icon=folium.Icon(color=color, icon='trash', prefix='fa')
            ).add_to(m)

    # Update session state with only pending requests
    st.session_state.active_requests = updated_requests

    # Add click functionality
    m.add_child(folium.LatLngPopup())

    return m


def main():
   
    # Apply background color to the main container
    st.markdown(
        """
        <style>
            div.block-container {padding-top: 1rem;}
            .stProgress > div > div > div > div {
                background-color: #4CAF50;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Show page loading progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        # Simulate some work
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    
    progress_bar.empty()  # Remove progress bar after loading

    # Title with color and spacing
    st.title("🌍 BBMP Waste Management System - SG Palya")

    # Sidebar menu with emojis for better navigation
    menu = ["🏠 Home", "📦 Request Pickup", "🚛 Track Vehicles", "♻️ Waste Classification"]
    choice = st.sidebar.radio("📌 Select an option", menu)

    # Function to create colored info boxes
    def info_box(text, emoji, color):
        st.markdown(
            f'<div style="background-color:{color}; padding:10px; border-radius:8px; font-size:18px; color:white;">{emoji} {text}</div>',
            unsafe_allow_html=True,
        )

    # Home Page
    if choice == "🏠 Home":
        st.subheader("🌱 **Welcome to BBMP Waste Management System**")
        st.write("### Our smart waste management system helps keep **SG Palya clean!**")

        # Create two columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            info_box("Request waste pickup through our app", "📱", "#2E8B57")
            st.write("")  # Add spacing
            info_box("Track collection vehicles in real-time", "🚛", "#1E90FF")

        with col2:
            info_box("AI-powered waste classification", "🔍", "#FF8C00")
            st.write("")
            info_box("Contribute to smarter waste management", "📊", "#A52A2A")

        # Add system status with progress bars
        st.subheader("System Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Active Vehicles", value="8")
            st.progress(80)  # 8/10 vehicles active
            
        with col2:
            st.metric(label="Collection Efficiency", value="85%")
            st.progress(85)
            
        with col3:
            st.metric(label="Request Completion", value="92%")
            st.progress(92)
    
    elif choice == "📦 Request Pickup":
        st.subheader("Request Waste Pickup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Your Name")
            contact = st.text_input("Contact Number")
        
            lat = None
            lon = None
            
            lat = st.number_input("Latitude", value=SG_PALYA_CENTER[0], format="%.6f")
            lon = st.number_input("Longitude", value=SG_PALYA_CENTER[1], format="%.6f")

            # Waste Type Selection
            waste_type = st.selectbox("♻️ Waste Category", 
                                    ["Bio-waste", "Paper/Cardboard", "Medical waste", "Glass", "Plastic"])

            # Submit Pickup Request
            if st.button("🚛 Submit Request"):
                with st.spinner("Processing your request..."):
                    data = {
                        "user": username if username else "anonymous",
                        "location": {"latitude": lat, "longitude": lon},
                        "waste_type": waste_type,
                        "status": "pending"
                    }

                    try:
                        response = requests.post(f"{API_BASE_URL}/request", json=data)
                        if response.status_code == 200:
                            st.success("✅ Pickup Request Submitted Successfully!")
                        else:
                            st.error("❌ Error submitting request")
                    except:
                        st.success("✅ Pickup Request Submitted Successfully! (Demo Mode)")
                        if "active_requests" not in st.session_state:
                            st.session_state.active_requests = []
                        st.session_state.active_requests.append(data)
                
                    # Show estimated pickup time
                    st.info("🕒 Estimated pickup time: Within 1-2 hours")
                    
                    # Show a progress bar for request handling
                    request_progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        request_progress.progress(i + 1)

            with col2:
                st.write("Your Location")
                loc_map = folium.Map(location=[lat if lat else SG_PALYA_CENTER[0], 
                                           lon if lon else SG_PALYA_CENTER[1]], 
                                 zoom_start=16)
                folium.Marker([lat if lat else SG_PALYA_CENTER[0], 
                            lon if lon else SG_PALYA_CENTER[1]], 
                          tooltip="Your Location").add_to(loc_map)
                st_folium(loc_map, width=400, height=300)
    

    elif choice == "🚛 Track Vehicles":
            st.subheader("🚛 Live Vehicle Tracking in SG Palya")

            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Refresh Data"):
                    st.session_state.last_update_time = time.time()  # Reset timer on manual refresh
                    st.rerun()

                # Route progress bar
                st.write("📍 Route Completion Progress")
                route_idx = st.session_state.current_route_index
                route_progress = st.session_state.route_progress
                total_progress = (route_idx + route_progress / 100) / len(SG_PALYA_ROUTE) * 100
                st.progress(int(total_progress))

            map_container = st.container()
            requests_container = st.container()

            col1, col2 = st.columns([3, 1])
            with col1:
                with st.spinner("Loading map..."):
                    m = create_map()
                    map_data = st_folium(m, width=700, height=500)

            with col2:
                with requests_container:
                    st.write("📝 Active Pickup Requests")
                    if not st.session_state.active_requests:
                        st.info("No active requests")
                    else:
                        for i, request in enumerate(st.session_state.active_requests):
                            st.write(f"🆔 Request #{i+1}")
                            st.write(f"Type: {request.get('waste_type', 'Unknown')}")
                            st.write(f"Status: {request.get('status', 'pending')}")
                            st.progress({"pending": 25, "approved": 50, "in_progress": 75, "completed": 100}.get(request.get('status', 'pending')))
                            st.write("---")

            # Truck movement logic
            if st.session_state.animation_running:
                current_time = time.time()
                if current_time - st.session_state.last_update_time > 5:
                    st.session_state.last_update_time = current_time

                    # Move truck forward along the route
                    if st.session_state.current_route_index < len(SG_PALYA_ROUTE) - 1:
                        st.session_state.current_route_index += 1
                        st.session_state.route_progress = 0  # Reset progress for new segment

                    # Check and cancel requests if the truck is near them
                    check_nearby_requests()

                    # Gradual movement update
                    for _ in range(1): # change to 1 to move 200m every 5 seconds.
                        time.sleep(0.5)  # Simulate movement delay
                        st.session_state.route_progress += 20  # Increase progress in steps
                        st.rerun()

        
    elif choice == "♻️ Waste Classification":
        st.subheader("AI Waste Classification")
        
        st.write("Our system uses computer vision to classify waste into categories:")
        st.write("• Bio-waste • Paper/Cardboard • Medical waste • Glass • Plastic")
        
        uploaded_file = st.file_uploader("Upload an image of waste for classification", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Show loading spinner while processing image
            with st.spinner("Loading image..."):
                image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process image with YOLO
            with st.spinner("🧠 Analyzing waste with AI..."):
                # Show a progress bar for AI processing
                analysis_progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)  # Simulate AI processing time
                    analysis_progress.progress(i + 1)
                
                detected_image, detections = detect_waste(image)
            
            with col2:
                st.image(detected_image, caption="Analyzed Waste", use_column_width=True)
                
                # Show detection results
                st.write("Classification Results:")
                if not detections:
                    st.write("No waste detected in image")
                else:
                    for detection in detections:
                        waste_type = detection["waste_type"]
                        confidence = detection["confidence"]
                        
                        # Show confidence as a progress bar
                        st.write(f"• {waste_type}")
                        st.progress(confidence)
                        st.write(f"  Confidence: {confidence:.2f}")
                        
                    # Add option to create pickup request based on detection
                    if st.button("Request Pickup for this Waste"):
                        with st.spinner("Creating pickup request..."):
                            # Determine primary waste type from detections
                            if detections:
                                primary_waste = max(detections, key=lambda x: x["confidence"])["waste_type"]
                                
                                # For demo, add a request at SG Palya with slight variation
                                new_request = {
                                    "user": "app_user",
                                    "location": {
                                        "latitude": SG_PALYA_CENTER[0] + random.uniform(-0.001, 0.001),
                                        "longitude": SG_PALYA_CENTER[1] + random.uniform(-0.001, 0.001)
                                    },
                                    "waste_type": primary_waste,
                                    "status": "pending"
                                }
                                
                                st.session_state.active_requests.append(new_request)
                                
                                # Show request progress
                                request_progress = st.progress(0)
                                for i in range(100):
                                    time.sleep(0.01)
                                    request_progress.progress(i + 1)
                                
                                st.success(f"Pickup request submitted for {primary_waste}")

if __name__ == "__main__":
    main()