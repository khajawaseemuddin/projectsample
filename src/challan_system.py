import os
import json
from datetime import datetime
import logging
from config import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/challan_generation.log'),
        logging.StreamHandler()
    ]
)

class ChallanGenerator:
    def __init__(self):
        # Create output directory
        os.makedirs(os.path.join(OUTPUT_DIR, "challans"), exist_ok=True)
        
        # Load data
        self.speed_data = self.load_speed_data()
        self.plate_data = self.load_plate_data()
        
    def load_speed_data(self):
        """Load speed data from JSON file"""
        try:
            with open(os.path.join(OUTPUT_DIR, 'speeding', 'speed_data.json'), 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading speed data: {str(e)}")
            return {'vehicle_speeds': {}}
            
    def load_plate_data(self):
        """Load plate data from JSON file"""
        try:
            with open(os.path.join(OUTPUT_DIR, 'plates', 'plate_data.json'), 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading plate data: {str(e)}")
            return {}
            
    def calculate_fine(self, speed):
        """Calculate fine based on speed with improved structure"""
        if speed <= BASE_SPEED_LIMIT:
            return 0
            
        excess_speed = speed - BASE_SPEED_LIMIT
        if excess_speed <= 20:
            return BASE_FINE
        elif excess_speed <= 40:
            return BASE_FINE * 2
        elif excess_speed <= 60:
            return BASE_FINE * 4
        else:
            return BASE_FINE * 8
            
    def generate_challan(self, vehicle_id):
        """Generate challan for a specific vehicle with improved template"""
        try:
            if vehicle_id not in self.plate_data:
                logging.warning(f"No plate data found for vehicle {vehicle_id}")
                return
                
            vehicle_data = self.plate_data[vehicle_id]
            if not vehicle_data.get('plates'):
                logging.warning(f"No plates found for vehicle {vehicle_id}")
                return
                
            # Get the best plate detection (highest confidence)
            best_plate = max(vehicle_data['plates'], key=lambda x: x.get('confidence', 0))
            
            # Create challan template
            template = Image.new('RGB', (800, 1000), 'white')
            draw = ImageDraw.Draw(template)
            
            # Load font
            try:
                title_font = ImageFont.truetype("arial.ttf", 24)
                text_font = ImageFont.truetype("arial.ttf", 16)
            except:
                title_font = ImageFont.load_default()
                text_font = ImageFont.load_default()
            
            # Add header
            draw.text((50, 50), "TRAFFIC CHALLAN", fill='black', font=title_font)
            
            # Add vehicle details
            y_offset = 100
            details = [
                f"Vehicle ID: {vehicle_id}",
                f"License Plate: {best_plate.get('plate_text', 'Not Recognized')}",
                f"Average Speed: {vehicle_data.get('average_speed', 0):.1f} km/h",
                f"Fine Amount: Rs. {self.calculate_fine(vehicle_data.get('average_speed', 0))}",
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Location: {LOCATION}"
            ]
            
            for detail in details:
                draw.text((50, y_offset), detail, fill='black', font=text_font)
                y_offset += 30
                
            # Add plate image
            try:
                plate_img = Image.open(best_plate['plate_path'])
                plate_img = plate_img.resize((200, 100))
                template.paste(plate_img, (50, y_offset))
            except Exception as e:
                logging.error(f"Error adding plate image: {str(e)}")
                
            # Save challan
            challan_path = os.path.join(OUTPUT_DIR, "challans", f"challan_{vehicle_id}.jpg")
            template.save(challan_path)
            
            logging.info(f"Generated challan for vehicle {vehicle_id}")
            
        except Exception as e:
            logging.error(f"Error generating challan for vehicle {vehicle_id}: {str(e)}")
            
    def generate_all_challans(self):
        """Generate challans for all vehicles with speed violations"""
        try:
            for vehicle_id in self.plate_data:
                self.generate_challan(vehicle_id)
                
            logging.info("Generated all challans successfully")
            
        except Exception as e:
            logging.error(f"Error generating all challans: {str(e)}")
            
    def export_challan_data(self):
        """Export challan data to JSON file"""
        try:
            challan_data = {}
            for vehicle_id, data in self.plate_data.items():
                challan_data[vehicle_id] = {
                    'plate_text': data['plates'][0].get('plate_text', '') if data['plates'] else '',
                    'average_speed': data.get('average_speed', 0),
                    'fine_amount': self.calculate_fine(data.get('average_speed', 0)),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
            with open(os.path.join(OUTPUT_DIR, 'challans', 'challan_data.json'), 'w') as f:
                json.dump(challan_data, f, indent=4)
                
            logging.info("Exported challan data successfully")
            
        except Exception as e:
            logging.error(f"Error exporting challan data: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Challan Generation System')
    parser.add_argument('--vehicle-id', type=str, help='Specific vehicle ID to generate challan for')
    args = parser.parse_args()
    
    generator = ChallanGenerator()
    
    if args.vehicle_id:
        # Generate challan for specific vehicle
        generator.generate_challan(args.vehicle_id)
    else:
        # Generate all challans
        generator.generate_all_challans()
        
    # Export challan data
    generator.export_challan_data()
    
    logging.info("Challan generation completed!")

if __name__ == '__main__':
    main() 