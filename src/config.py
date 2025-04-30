# Model configuration
MODEL_PATH = "yolov8n.pt"  # Path to YOLO model weights
VEHICLE_CLASSES = [2, 3, 5, 7]  # Class IDs for vehicles (car, motorcycle, bus, truck)
VEHICLE_CONFIDENCE = 0.5  # Minimum confidence threshold for vehicle detection

# Speed detection parameters
FPS = 30  # Frames per second
PIXELS_PER_METER = 10  # Pixels per meter for speed calculation
ROI_POINTS = [(0, 0), (640, 0), (640, 384), (0, 384)]  # Region of interest points
TRACK_HISTORY_LENGTH = 30  # Number of frames to keep in track history
SPEED_HISTORY_LENGTH = 10  # Number of speed measurements to keep
MAX_SCREENSHOTS_PER_VEHICLE = 3  # Maximum number of screenshots to capture per vehicle

# Output configuration
OUTPUT_DIR = "outputs"  # Base directory for output files
VIDEO_PATH = "uploads/short.mp4"  # Path to input video file in uploads directory 