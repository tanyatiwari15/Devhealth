from flask import Blueprint, render_template, Response, jsonify
import cv2
from .posture.detector import PostureDetector
import threading

routes = Blueprint('routes', __name__)
detector = PostureDetector()
camera = None
current_metrics = {
    "neck_angle": 0,
    "torso_angle": 0,
    "posture": "checking"
}
metrics_lock = threading.Lock()

def update_metrics(metrics):
    global current_metrics
    with metrics_lock:
        current_metrics = metrics

def gen_frames():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame
        processed_frame, metrics = detector.process_frame(frame)
        update_metrics(metrics)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@routes.route('/')
def index():
    return render_template('index.html')

@routes.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@routes.route('/metrics')
def get_metrics():
    with metrics_lock:
        return jsonify(current_metrics)

@routes.route('/start_camera')
def start_camera():
    global camera
    if camera is None:
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return jsonify({'success': False, 'error': 'Failed to open camera'}), 500
            return jsonify({'success': True}), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': True}), 200

@routes.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'success': True}), 200