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
from streamlit_folium import st_folium




API_BASE_URL = "http://127.0.0.1:8000"


SG_PALYA_CENTER = (12.9352, 77.6095)


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


if 'vehicle_location' not in st.session_state:
    st.session_state.vehicle_location = SG_PALYA_ROUTE[0]

if 'active_requests' not in st.session_state:
    st.session_state.active_requests = []

if 'completed_requests' not in st.session_state:  
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


@st.cache_resource
def load_custom_model():
    model_path = r"weights/best.pt"  
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return YOLO(model_path)

try:
    with st.spinner("Loading AI model..."):
        model = load_custom_model()
except FileNotFoundError:
    st.warning("Model file not found. Using mock detection for demonstration.")
    model = None


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
  
    image = np.array(image)

   
    image = cv2.resize(image, (640, 640))

    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  
    image = cv2.filter2D(image, -1, sharpening_kernel)

  
    image_pil = Image.fromarray(image)

    enhancer = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer.enhance(1.2) 


    enhanced_image = np.array(image_pil)

    enhanced_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)

    return enhanced_image

def detect_waste(image):
    """Run YOLOv8 on the uploaded image and classify waste types with preprocessing."""
    if model is None:
       
        waste_types = list(WASTE_CLASSES.values())
        detections = [{"waste_type": random.choice(waste_types), "confidence": random.uniform(0.7, 0.95)}]
        return np.array(image), detections

   
    processed_image = preprocess_image(image)

    results = model.predict(processed_image, conf=0.2)

    annotated_image = results[0].plot()

    detections = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0]) 
            class_id = int(box.cls[0])  

            waste_type = WASTE_CLASSES.get(class_id, "Unknown")

            detections.append({
                "waste_type": waste_type,
                "confidence": confidence
            })

    return annotated_image, detections

def get_active_requests():
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



def check_nearby_requests():
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

            if distance < 100:
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/complete_request",
                        json={"request_id": request.get("id", ""), "status": "completed"}
                    )
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


def create_map():
    m = folium.Map(location=SG_PALYA_CENTER, zoom_start=16)
    
    folium.Marker(
        st.session_state.vehicle_location,
        tooltip="🚛 Waste Collection Truck",
        icon=folium.Icon(color='blue', icon='truck', prefix='fa')
    ).add_to(m)
    
    folium.PolyLine(
        SG_PALYA_ROUTE,
        color='blue',
        weight=3,
        opacity=0.8
    ).add_to(m)
    
    updated_requests = []
    truck_location = st.session_state.vehicle_location
    
    for request in st.session_state.active_requests:
        request_location = (
            request.get("location", {}).get("latitude", 12.9352 + random.uniform(-0.003, 0.003)),
            request.get("location", {}).get("longitude", 77.6095 + random.uniform(-0.003, 0.003))
        )
        waste_type = request.get("waste_type", "Unknown")
        
        color_map = {
            "Bio-waste": "green",
            "Paper/Cardboard": "blue",
            "Plastic": "red",
            "Glass": "purple",
            "Medical waste": "orange"
        }
        color = color_map.get(waste_type, "gray")
        
        distance = geodesic(truck_location, request_location).meters
        
        if distance < 50:
            st.success(f"✅ Pickup request completed at {request_location}!")
        else:
            updated_requests.append(request)
            folium.Marker(
                request_location,
                tooltip=f"🗑️ Pickup Request: {waste_type} ({round(distance, 2)}m away)",
                icon=folium.Icon(color=color, icon='trash', prefix='fa')
            ).add_to(m)
    
    st.session_state.active_requests = updated_requests
    
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

    
    progress_bar = st.progress(0)
    for i in range(100):
        # Simulate some work
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    
    progress_bar.empty()  

  
    st.title("🌍 BBMP Waste Management System - SG Palya")

   
    menu = ["🏠 Home", "📦 Request Pickup", "🚛 Track Vehicles", "♻️ Waste Classification"]
    choice = st.sidebar.radio("📌 Select an option", menu)

    
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

           


        
    elif choice == "♻️ Waste Classification":
        st.subheader("AI Waste Classification")
        
        st.write("Our system uses computer vision to classify waste into categories:")
        st.write("• Bio-waste • Paper/Cardboard • Medical waste • Glass • Plastic")
        
        uploaded_file = st.file_uploader("Upload an image of waste for classification", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            
            with st.spinner("Loading image..."):
                image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
           
            with st.spinner("🧠 Analyzing waste with AI..."):
               
                analysis_progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.02) 
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