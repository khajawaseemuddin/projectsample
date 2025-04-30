# File: src/test_plate_darknet.py
import sys
import os
import time
from datetime import datetime

# Add both the root directory and src directory to the path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

from tool.darknet2pytorch import Darknet
import cv2
import torch
import numpy as np

def create_directories():
    """Create necessary directories for storing data"""
    os.makedirs("outputs/cars", exist_ok=True)
    os.makedirs("outputs/plates", exist_ok=True)
    os.makedirs("outputs/detections", exist_ok=True)
    print("Directories created")

def preprocess_image(image, input_size=416):
    img = cv2.resize(image, (input_size, input_size))
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)
    return img

def detect_cars(frame, model, confidence_threshold=0.5):
    """Detect cars in the frame"""
    img_tensor = preprocess_image(frame)
    with torch.no_grad():
        detections = model(img_tensor)
    
    car_boxes = []
    if detections is not None:
        for detection in detections:
            if detection is None:
                continue
            if detection.dim() != 2 or detection.size(1) < 5:
                continue
                
            boxes = detection[..., :4].cpu().numpy()
            scores = detection[..., 4].cpu().numpy()
            
            for i, score in enumerate(scores):
                if score > confidence_threshold:
                    if i >= len(boxes):
                        continue
                    car_boxes.append(boxes[i])
    
    return car_boxes

def save_car_image(frame, box, frame_count, timestamp):
    """Save car image and return the path"""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = int(x1 * width / 416)
    y1 = int(y1 * height / 416)
    x2 = int(x2 * width / 416)
    y2 = int(y2 * height / 416)
    
    # Ensure coordinates are within frame bounds
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    car_img = frame[y1:y2, x1:x2]
    if car_img.size == 0:
        return None
        
    # Save car image
    car_path = f"outputs/cars/car_{frame_count:03d}_{timestamp}.jpg"
    cv2.imwrite(car_path, car_img)
    
    # Save frame with detection box
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    detection_path = f"outputs/detections/detection_{frame_count:03d}_{timestamp}.jpg"
    cv2.imwrite(detection_path, display_frame)
    
    return car_path

def analyze_car_image(car_path, plate_model):
    """Analyze saved car image for license plate"""
    if not os.path.exists(car_path):
        return None
        
    car_img = cv2.imread(car_path)
    if car_img is None:
        return None
        
    # Focus on lower portion of the car image (where rear plate typically is)
    height, width = car_img.shape[:2]
    lower_portion = car_img[int(height*0.7):, :]  # Take bottom 30% of the car image
    
    # Preprocess for plate detection
    plate_tensor = preprocess_image(lower_portion)
    
    with torch.no_grad():
        plate_detections = plate_model(plate_tensor)
    
    if plate_detections is not None:
        for detection in plate_detections:
            if detection is None:
                continue
            if detection.dim() != 2 or detection.size(1) < 5:
                continue
                
            boxes = detection[..., :4].cpu().numpy()
            scores = detection[..., 4].cpu().numpy()
            
            for i, score in enumerate(scores):
                if score > 0.3:  # Confidence threshold for plate detection
                    if i >= len(boxes):
                        continue
                        
                    x1, y1, x2, y2 = boxes[i]
                    x1 = int(x1 * width / 416)
                    y1 = int(y1 * height / 416) + int(height * 0.7)
                    x2 = int(x2 * width / 416)
                    y2 = int(y2 * height / 416) + int(height * 0.7)
                    
                    # Ensure coordinates are within bounds
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    plate_img = car_img[y1:y2, x1:x2]
                    if plate_img.size == 0:
                        continue
                        
                    # Save plate image
                    plate_path = f"outputs/plates/plate_{os.path.basename(car_path)}"
                    cv2.imwrite(plate_path, plate_img)
                    return plate_path
    
    return None

def main():
    try:
        print("Initializing...")
        create_directories()
        
        # Load models
        print("Loading models...")
        car_model = Darknet("weights/yolov4-tiny.cfg")
        car_model.load_state_dict(torch.load("weights/yolov4-tiny.pt"))
        car_model.eval()
        
        plate_model = Darknet("weights/yolov4-tiny.cfg")  # Using same model for now
        plate_model.load_state_dict(torch.load("weights/yolov4-tiny.pt"))
        plate_model.eval()
        print("Models loaded")
        
        # Open video file
        video_path = "videos/pexels_traffic.mp4"
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video properties: {width}x{height} @ {fps} fps")
        
        frame_count = 0
        last_car_time = 0
        car_detection_interval = 1.0  # Minimum time between car detections (seconds)
        
        print("Processing video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
                
            current_time = time.time()
            
            # Only process every nth frame to reduce processing load
            if frame_count % 5 == 0:  # Process every 5th frame
                # Detect cars
                car_boxes = detect_cars(frame, car_model)
                
                # If cars detected and enough time has passed since last detection
                if car_boxes and (current_time - last_car_time) > car_detection_interval:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    for box in car_boxes:
                        car_path = save_car_image(frame, box, frame_count, timestamp)
                        if car_path:
                            print(f"Saved car image: {car_path}")
                            # Analyze car image for license plate
                            plate_path = analyze_car_image(car_path, plate_model)
                            if plate_path:
                                print(f"Detected plate: {plate_path}")
                    
                    last_car_time = current_time
            
            frame_count += 1
            
            # Display progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        print(f"Processing complete. Total frames: {frame_count}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    finally:
        if cap is not None:
            cap.release()
        print("Cleanup complete")

if __name__ == "__main__":
    main()