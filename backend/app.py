from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import base64
from collections import deque
import threading
import queue
from preprocess import process_image

app = Flask(__name__)

# Configure CUDA if available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLO model
model = YOLO('best.pt')
model.to(DEVICE)

# Temporal smoothing for predictions
class PredictionSmoother:
    def __init__(self, window_size=5, threshold=0.4):
        self.window_size = window_size
        self.threshold = threshold
        self.predictions = deque(maxlen=window_size)
        
    def update(self, status, confidence):
        self.predictions.append((status, confidence))
        
        if len(self.predictions) < self.window_size:
            return status, confidence
            
        # Count recent predictions
        bad_count = sum(1 for s, c in self.predictions if s == 'bad' and c > self.threshold)
        good_count = sum(1 for s, c in self.predictions if s == 'good' and c > self.threshold)
        
        # Calculate average confidence
        avg_conf = np.mean([c for _, c in self.predictions])
        
        # Determine status based on majority
        if bad_count > good_count:
            return 'bad', avg_conf
        elif good_count > bad_count:
            return 'good', avg_conf
        else:
            return 'unknown', avg_conf

smoother = PredictionSmoother(window_size=3, threshold=0.3)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        image_data = data['frame']
        
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        img_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        # Preprocess the frame to get stick figure
        processed_frame = process_image(image=frame)
        if processed_frame is None:
            return jsonify({
                'status': 'unknown',
                'confidence': 0.0,
                'raw_confidence': 0.0
            })
        
        # Process with YOLO using lower confidence threshold
        results = model(processed_frame, conf=0.3, device=DEVICE)
        
        if len(results) > 0:
            result = results[0]
            if len(result.boxes) > 0:
                # Get all predictions above threshold
                predictions = []
                for box in result.boxes:
                    conf = float(box.conf)
                    cls = int(box.cls)
                    predictions.append((cls, conf))
                
                # Sort by confidence and get highest confidence prediction
                predictions.sort(key=lambda x: x[1], reverse=True)
                cls, conf = predictions[0]
                
                # Apply temporal smoothing
                status = 'bad' if cls == 1 else 'good'
                smoothed_status, smoothed_conf = smoother.update(status, conf)
                
                return jsonify({
                    'status': smoothed_status,
                    'confidence': smoothed_conf,
                    'raw_confidence': conf
                })
        
        smoothed_status, smoothed_conf = smoother.update('unknown', 0.0)
        return jsonify({
            'status': smoothed_status,
            'confidence': smoothed_conf,
            'raw_confidence': 0.0
        })
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'confidence': 0.0,
            'raw_confidence': 0.0
        }), 400

if __name__ == '__main__':
    app.run(debug=True)