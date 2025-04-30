import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, send_file
import json
from datetime import datetime
import logging
from config import *
from src.speed_detection import SpeedDetector
from src.license_plate_detection import LicensePlateDetector
from src.challan_system import ChallanGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/app.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

# Create necessary directories
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('videos', exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'speeding'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'plates'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'challans'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'detections'), exist_ok=True)

# Initialize components
speed_detector = SpeedDetector()
plate_detector = LicensePlateDetector()
challan_generator = ChallanGenerator()

@app.route('/')
def index():
    """Render the main dashboard"""
    return render_template('index.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start the speed and license plate detection process"""
    try:
        # Get video file from request
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Save video file
        video_path = os.path.join('videos', video_file.filename)
        video_file.save(video_path)
        
        # Update config
        global VIDEO_PATH
        VIDEO_PATH = video_path
        
        # Start detection process
        speed_detector.process_video(video_path)
        plate_detector.process_speeding_vehicles()
        
        return jsonify({'message': 'Detection process started successfully'})
        
    except Exception as e:
        logging.error(f"Error in detection process: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_challans', methods=['POST'])
def generate_challans():
    """Generate challans for detected vehicles"""
    try:
        vehicle_id = request.json.get('vehicle_id')
        
        if vehicle_id:
            # Generate specific challan
            challan_generator.generate_challan(vehicle_id)
        else:
            # Generate all challans
            challan_generator.generate_all_challans()
            
        # Export challan data
        challan_generator.export_challan_data()
        
        return jsonify({'message': 'Challans generated successfully'})
        
    except Exception as e:
        logging.error(f"Error generating challans: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_detections')
def get_detections():
    """Get all detected vehicles and their data"""
    try:
        # Initialize empty data structures
        speed_data = {'vehicle_speeds': {}}
        plate_data = {}
        
        # Try to load speed data if it exists
        speed_data_path = os.path.join(OUTPUT_DIR, 'speeding', 'speed_data.json')
        if os.path.exists(speed_data_path):
            with open(speed_data_path, 'r') as f:
                speed_data = json.load(f)
                
        # Try to load plate data if it exists
        plate_data_path = os.path.join(OUTPUT_DIR, 'plates', 'plate_data.json')
        if os.path.exists(plate_data_path):
            with open(plate_data_path, 'r') as f:
                plate_data = json.load(f)
                
        # Combine data
        detections = []
        for vehicle_id, data in plate_data.items():
            speed_info = speed_data['vehicle_speeds'].get(vehicle_id, {})
            detections.append({
                'vehicle_id': vehicle_id,
                'plate_text': data['plates'][0].get('plate_text', '') if data['plates'] else '',
                'average_speed': data.get('average_speed', 0),
                'max_speed': speed_info.get('max_speed', 0),
                'fine_amount': challan_generator.calculate_fine(data.get('average_speed', 0)),
                'plate_image': data['plates'][0].get('plate_path', '') if data['plates'] else '',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
        return jsonify({'detections': detections})
        
    except Exception as e:
        logging.error(f"Error getting detections: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_challan/<vehicle_id>')
def get_challan(vehicle_id):
    """Get generated challan for a specific vehicle"""
    try:
        challan_path = os.path.join(OUTPUT_DIR, 'challans', f'challan_{vehicle_id}.jpg')
        if os.path.exists(challan_path):
            return send_file(challan_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Challan not found'}), 404
            
    except Exception as e:
        logging.error(f"Error getting challan: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create index.html template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traffic Violation Detection System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .dashboard-card {
            transition: transform 0.2s;
        }
        .dashboard-card:hover {
            transform: translateY(-5px);
        }
        .detection-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .status-message {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .status-info {
            background-color: #e7f3fe;
            border-left: 6px solid #2196F3;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container">
            <span class="navbar-brand mb-0 h1">Traffic Violation Detection System</span>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-md-4">
                <div class="card dashboard-card mb-4">
                    <div class="card-body">
                        <h5 class="card-title"><i class="fas fa-video"></i> Video Upload</h5>
                        <form id="uploadForm" enctype="multipart/form-data">
                            <div class="mb-3">
                                <input type="file" class="form-control" id="videoFile" accept="video/*" required>
                            </div>
                            <button type="submit" class="btn btn-primary">Start Detection</button>
                        </form>
                    </div>
                </div>
            </div>
            
            <div class="col-md-8">
                <div class="card dashboard-card">
                    <div class="card-body">
                        <h5 class="card-title"><i class="fas fa-list"></i> Detections</h5>
                        <div id="statusMessage" class="status-message status-info">
                            <i class="fas fa-info-circle"></i> No speeding vehicles detected yet. Please upload a video and start detection.
                        </div>
                        <div class="detection-list" id="detectionList">
                            <!-- Detections will be loaded here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Handle video upload
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData();
            formData.append('video', document.getElementById('videoFile').files[0]);
            
            try {
                const statusMessage = document.getElementById('statusMessage');
                statusMessage.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing video...';
                statusMessage.className = 'status-message status-info';
                
                const response = await fetch('/start_detection', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (response.ok) {
                    statusMessage.innerHTML = '<i class="fas fa-check-circle"></i> Detection process started successfully';
                    statusMessage.className = 'status-message status-success';
                    loadDetections();
                } else {
                    statusMessage.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error: ${data.error}`;
                    statusMessage.className = 'status-message status-error';
                }
            } catch (error) {
                const statusMessage = document.getElementById('statusMessage');
                statusMessage.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error: ${error}`;
                statusMessage.className = 'status-message status-error';
            }
        });

        // Load detections
        async function loadDetections() {
            try {
                const response = await fetch('/get_detections');
                const data = await response.json();
                
                if (response.ok) {
                    const detectionList = document.getElementById('detectionList');
                    const statusMessage = document.getElementById('statusMessage');
                    
                    if (data.detections.length === 0) {
                        statusMessage.innerHTML = '<i class="fas fa-info-circle"></i> No speeding vehicles detected yet. Please upload a video and start detection.';
                        statusMessage.className = 'status-message status-info';
                        detectionList.innerHTML = '';
                    } else {
                        statusMessage.innerHTML = `<i class="fas fa-check-circle"></i> Found ${data.detections.length} speeding vehicles`;
                        statusMessage.className = 'status-message status-success';
                        
                        detectionList.innerHTML = '';
                        data.detections.forEach(detection => {
                            const card = document.createElement('div');
                            card.className = 'card mb-2';
                            card.innerHTML = `
                                <div class="card-body">
                                    <h6 class="card-title">Vehicle ID: ${detection.vehicle_id}</h6>
                                    <p class="card-text">
                                        Plate: ${detection.plate_text}<br>
                                        Speed: ${detection.average_speed.toFixed(1)} km/h<br>
                                        Fine: Rs. ${detection.fine_amount}
                                    </p>
                                    <button class="btn btn-sm btn-primary" onclick="generateChallan('${detection.vehicle_id}')">
                                        Generate Challan
                                    </button>
                                </div>
                            `;
                            detectionList.appendChild(card);
                        });
                    }
                }
            } catch (error) {
                console.error('Error loading detections:', error);
                const statusMessage = document.getElementById('statusMessage');
                statusMessage.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error loading detections: ${error}`;
                statusMessage.className = 'status-message status-error';
            }
        }

        // Generate challan
        async function generateChallan(vehicleId) {
            try {
                const response = await fetch('/generate_challans', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ vehicle_id: vehicleId })
                });
                
                const data = await response.json();
                if (response.ok) {
                    window.open(`/get_challan/${vehicleId}`, '_blank');
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error: ' + error);
            }
        }

        // Load detections on page load
        loadDetections();
    </script>
</body>
</html>
        ''')
    
    app.run(debug=True)
