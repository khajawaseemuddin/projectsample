import subprocess
import threading
import time
import os
from flask import Flask
from app import app

def run_speed_detection():
    """Run the speed detection script"""
    subprocess.run(['python', 'src/speed_detection.py'])

def run_web_app():
    """Run the Flask web application"""
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('outputs/speeding', exist_ok=True)
    os.makedirs('src/static/speeding', exist_ok=True)
    
    # Create symbolic link to outputs in static directory
    if not os.path.exists('src/static/speeding'):
        os.symlink('../../outputs/speeding', 'src/static/speeding')
    
    # Start speed detection in a separate thread
    speed_thread = threading.Thread(target=run_speed_detection)
    speed_thread.start()
    
    # Wait a moment for the speed detection to initialize
    time.sleep(2)
    
    # Start the web application
    run_web_app() 