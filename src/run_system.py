import subprocess
import time
import os
from flask import Flask
from app import app

def run_speed_detection():
    """Run the speed detection script"""
    print("Starting speed detection...")
    subprocess.run(['python', 'src/speed_detection.py'])
    print("Speed detection completed!")

def run_license_plate_detection():
    """Run the license plate detection script"""
    print("Starting license plate detection...")
    subprocess.run(['python', 'src/license_plate_detection.py'])
    print("License plate detection completed!")

def run_challan_system():
    """Run the challan generation system"""
    print("Starting challan generation...")
    subprocess.run(['python', 'src/challan_system.py'])
    print("Challan generation completed!")

def run_web_app():
    """Run the Flask web application"""
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('outputs/speeding', exist_ok=True)
    os.makedirs('outputs/plates', exist_ok=True)
    os.makedirs('outputs/challans', exist_ok=True)
    os.makedirs('src/static/speeding', exist_ok=True)
    
    # Create symbolic link to outputs in static directory
    if not os.path.exists('src/static/speeding'):
        os.symlink('../../outputs/speeding', 'src/static/speeding')
    
    print("Starting Traffic Violation Detection System...")
    
    # Step 1: Run speed detection
    run_speed_detection()
    
    # Step 2: Run license plate detection
    run_license_plate_detection()
    
    # Step 3: Generate challans
    run_challan_system()
    
    print("All processing completed! Starting web interface...")
    
    # Step 4: Start web interface
    run_web_app() 