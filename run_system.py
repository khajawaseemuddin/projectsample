import subprocess
import sys
import os
import time

def main():
    # Create necessary directories
    os.makedirs('src/templates', exist_ok=True)
    os.makedirs('outputs/detections', exist_ok=True)
    os.makedirs('outputs/speeding', exist_ok=True)

    # Start the speed detection in a separate process
    speed_detection_process = subprocess.Popen([sys.executable, 'src/speed_detection.py'])
    
    # Wait a moment for the speed detection to initialize
    time.sleep(2)
    
    # Start the web server
    web_server_process = subprocess.Popen([sys.executable, 'src/web_server.py'])
    
    try:
        # Wait for both processes
        speed_detection_process.wait()
        web_server_process.wait()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        speed_detection_process.terminate()
        web_server_process.terminate()
        print("\nShutting down...")

if __name__ == '__main__':
    main() 