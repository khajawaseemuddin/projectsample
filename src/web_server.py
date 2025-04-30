from flask import Flask, render_template, jsonify, send_file
import json
import os
import glob

app = Flask(__name__)

def get_detection_results():
    """Get speed detection results"""
    results_path = os.path.join('outputs', 'detections', 'detection_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {
        'total_frames': 0,
        'total_vehicles': 0,
        'speeding_vehicles': 0,
        'speeding_vehicle_ids': []
    }

def get_plate_data():
    """Get license plate data for speeding vehicles"""
    # Get speed detection results first to know which vehicles were speeding
    results = get_detection_results()
    speeding_ids = results.get('speeding_vehicle_ids', [])
    
    # Process plate data only for speeding vehicles
    vehicle_details = {}
    
    # Read the speed data
    speed_data_path = os.path.join('outputs', 'speeding', 'speed_data.json')
    speed_data = {}
    if os.path.exists(speed_data_path):
        with open(speed_data_path, 'r') as f:
            speed_data = json.load(f)
            vehicle_speeds = speed_data.get('vehicle_speeds', {})
    
    # Read the plate detection results
    plate_results_path = os.path.join('outputs', 'plates', 'plate_detection_results.json')
    if os.path.exists(plate_results_path):
        with open(plate_results_path, 'r') as f:
            plate_data = json.load(f)
            all_vehicle_details = plate_data.get('vehicle_details', {})
            
            for vehicle_id in speeding_ids:
                str_vehicle_id = str(vehicle_id)
                # Get speed data for this vehicle
                vehicle_speed_data = vehicle_speeds.get(str_vehicle_id, {})
                avg_speed = float(vehicle_speed_data.get('average_speed', 0))
                
                if str_vehicle_id in all_vehicle_details:
                    # Get the plate detections for this vehicle
                    detections = all_vehicle_details[str_vehicle_id]
                    if detections:
                        # Use the detection with highest confidence
                        best_detection = max(detections, key=lambda x: x['confidence'])
                        vehicle_details[str_vehicle_id] = {
                            'plate_text': best_detection['plate_text'] or 'Unreadable',
                            'confidence': best_detection['confidence'],
                            'speed': avg_speed
                        }
                else:
                    vehicle_details[str_vehicle_id] = {
                        'plate_text': 'No plate detected',
                        'confidence': 0,
                        'speed': avg_speed
                    }
    
    return {
        'total_vehicles_processed': len(speeding_ids),
        'plates_detected': len(vehicle_details),
        'vehicle_details': vehicle_details
    }

@app.route('/')
def index():
    # Get detection results and plate data
    results = get_detection_results()
    plate_data = get_plate_data()
    
    return render_template('index.html', results=results, plate_data=plate_data)

@app.route('/api/results')
def get_results():
    results = get_detection_results()
    plate_data = get_plate_data()
    
    return jsonify({
        'speed_results': results,
        'plate_data': plate_data
    })

@app.route('/api/plates')
def get_plates():
    return jsonify(get_plate_data())

@app.route('/plates/<vehicle_id>')
def get_plate_image(vehicle_id):
    # Try both formats of filenames
    patterns = [
        os.path.join('outputs', 'plates', f'plate_vehicle_{vehicle_id}_*.jpg'),
        os.path.join('outputs', 'plates', f'plate_vehicle_{vehicle_id}_*kmh.jpg')
    ]
    
    for pattern in patterns:
        plate_files = glob.glob(pattern)
        if plate_files:
            return send_file(plate_files[0], mimetype='image/jpeg')
    
    return '', 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 