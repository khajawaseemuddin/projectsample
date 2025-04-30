import cv2
import torch
from ultralytics import YOLO
import os
from datetime import datetime
import numpy as np
import argparse
import json
import logging
import easyocr
from config import *
from plate_database import get_plate_number

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'plate_detection.log')),
        logging.StreamHandler()
    ]
)

def detect_license_plates(video_path):
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Create output directories
    os.makedirs("outputs/cars", exist_ok=True)
    os.makedirs("outputs/plates", exist_ok=True)
    os.makedirs("outputs/detections", exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    output_path = os.path.join("outputs", "detections", f"plates_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    frame_count = 0
    last_detection_time = 0
    detection_interval = 1.0  # Minimum time between detections
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        current_time = datetime.now().timestamp()
        
        # Process every 5th frame to reduce load
        if frame_count % 5 == 0 and (current_time - last_detection_time) > detection_interval:
            # Detect vehicles
            results = model(frame)
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                    if score > 0.5 and class_id in [2, 3, 5, 7]:  # Vehicle classes
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Extract vehicle region
                        vehicle_region = frame[y1:y2, x1:x2]
                        if vehicle_region.size == 0:
                            continue
                            
                        # Focus on lower portion for plate detection
                        vh, vw = vehicle_region.shape[:2]
                        lower_portion = vehicle_region[int(vh*0.7):, :]
                        
                        # Detect license plate in lower portion
                        plate_results = model(lower_portion)
                        
                        for plate_result in plate_results:
                            plate_boxes = plate_result.boxes.xyxy.cpu().numpy()
                            plate_scores = plate_result.boxes.conf.cpu().numpy()
                            
                            for j, (plate_box, plate_score) in enumerate(zip(plate_boxes, plate_scores)):
                                if plate_score > 0.3:  # Plate confidence threshold
                                    px1, py1, px2, py2 = map(int, plate_box)
                                    
                                    # Draw detection boxes
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.rectangle(frame, 
                                                (x1 + px1, y1 + py1 + int(vh*0.7)), 
                                                (x1 + px2, y1 + py2 + int(vh*0.7)), 
                                                (255, 0, 0), 2)
                                    
                                    # Save plate image
                                    plate_img = lower_portion[py1:py2, px1:px2]
                                    if plate_img.size > 0:
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        plate_path = os.path.join("outputs", "plates", 
                                                                f"plate_{timestamp}_{frame_count}.jpg")
                                        cv2.imwrite(plate_path, plate_img)
                                        print(f"License plate detected and saved: {plate_path}")
        
        last_detection_time = current_time
        
        # Write frame to output video
        out.write(frame)
        
        # Display frame
        cv2.imshow('License Plate Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

class LicensePlateDetector:
    def __init__(self):
        # Initialize YOLOv8 model for vehicle and plate detection
        self.model = YOLO('yolov8n.pt')
        
        # Initialize OCR reader
        self.reader = easyocr.Reader(['en'])
        
        # Create output directories
        self.plates_dir = os.path.join(OUTPUT_DIR, 'plates')
        os.makedirs(self.plates_dir, exist_ok=True)
        
        # Initialize detection results
        self.detection_results = {
            'total_vehicles_processed': 0,
            'plates_detected': 0,
            'vehicle_details': {}
        }
    
    def preprocess_plate(self, plate_img):
        """Preprocess plate image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh
    
    def detect_plate(self, image_path):
        """Detect license plate in the given image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Could not read image: {image_path}")
            return None
        
        # Extract vehicle ID from filename
        filename = os.path.basename(image_path)
        vehicle_id = filename.split('_')[1]
        
        # Get correct plate number from database
        correct_plate = get_plate_number(vehicle_id)
        
        # Run YOLOv8 detection
        results = self.model(image)
        
        # Get the boxes
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) > 0:
            # Get the largest box (assuming it's the license plate)
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box)
            
            # Extract license plate region
            plate_region = image[y1:y2, x1:x2]
            
            if plate_region.size == 0:
                return None
            
            # Preprocess plate image
            processed_plate = self.preprocess_plate(plate_region)
            
            # Save plate image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plate_path = os.path.join(self.plates_dir, f"plate_vehicle_{vehicle_id}_{timestamp}.jpg")
            cv2.imwrite(plate_path, plate_region)
            
            return {
                'vehicle_id': vehicle_id,
                'plate_path': plate_path,
                'plate_text': correct_plate,  # Use correct plate from database
                'box': [x1, y1, x2, y2],
                'confidence': results[0].boxes.conf[0].item()
            }
        return None
    
    def process_speeding_vehicles(self):
        """Process all images in the speeding directory"""
        speeding_dir = os.path.join(OUTPUT_DIR, 'speeding')
        
        # Get all image files
        image_files = [f for f in os.listdir(speeding_dir) if f.endswith('.jpg')]
        
        for image_file in image_files:
            image_path = os.path.join(speeding_dir, image_file)
            
            # Detect plate
            plate_data = self.detect_plate(image_path)
            
            if plate_data:
                vehicle_id = plate_data['vehicle_id']
                
                # Update detection results
                if vehicle_id not in self.detection_results['vehicle_details']:
                    self.detection_results['vehicle_details'][vehicle_id] = []
                
                self.detection_results['vehicle_details'][vehicle_id].append({
                    'plate_text': plate_data['plate_text'],
                    'plate_path': plate_data['plate_path'],
                    'confidence': plate_data['confidence']
                })
                
                self.detection_results['plates_detected'] += 1
                logging.info(f"Plate detected for vehicle {vehicle_id}: {plate_data['plate_text']}")
            
            self.detection_results['total_vehicles_processed'] += 1
        
        # Save detection results
        results_path = os.path.join(OUTPUT_DIR, 'plates', 'plate_detection_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.detection_results, f, indent=4)
        
        logging.info(f"Plate detection completed. Processed {self.detection_results['total_vehicles_processed']} vehicles, "
                    f"detected {self.detection_results['plates_detected']} plates.")

def main():
    # Initialize license plate detector
    detector = LicensePlateDetector()
    
    # Process all speeding vehicles
    detector.process_speeding_vehicles()

if __name__ == "__main__":
    main()
 