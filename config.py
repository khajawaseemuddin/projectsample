import os

# Paths
VIDEO_PATH = "uploads/short.mp4"
MODEL_PATH = "yolov8n.pt"
OUTPUT_DIR = "outputs"

# Speed Detection
SPEED_LIMIT = 80  # km/h
MIN_SPEED_FOR_CHALLAN = 80
MAX_SPEED_FOR_CHALLAN = 120
PIXELS_PER_METER = 10
FPS = 30
ROI_POINTS = [[100, 400], [700, 400], [800, 600], [0, 600]]  # Region of Interest

# Detection Thresholds
VEHICLE_CONFIDENCE = 0.5
PLATE_CONFIDENCE = 0.3
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Processing
FRAME_SKIP = 5  # Process every Nth frame
MAX_SCREENSHOTS_PER_VEHICLE = 3
SPEED_HISTORY_LENGTH = 100
TRACK_HISTORY_LENGTH = 30

# Challan System
BASE_FINE = 1000
FINE_PER_KM_OVER_LIMIT = 100
CHALLAN_TEMPLATE_PATH = "templates/challan_template.jpg"

# Create necessary directories
os.makedirs(os.path.join(OUTPUT_DIR, "speeding"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plates"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "challans"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "detections"), exist_ok=True)
os.makedirs("templates", exist_ok=True) 